# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn

from cvpods.layers import ShapeSpec
from cvpods.modeling.nn_utils.scale_grad import _ScaleGradient
from cvpods.structures import Boxes, Instances, pairwise_iou
from cvpods.utils import get_event_storage

from cvpods.modeling.box_regression import Box2BoxTransform
from cvpods.modeling.matcher import Matcher
from cvpods.modeling.poolers import ROIPooler
from cvpods.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference
from cvpods.modeling.roi_heads.roi_heads import StandardROIHeads

from mask_head import mask_rcnn_inference, mask_rcnn_loss

def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instance object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero(as_tuple=False).squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks

class CascadeDEQROIHeads(StandardROIHeads):
    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        self.num_cascade_stages  = len(cascade_ious)
        assert len(cascade_bbox_reg_weights) == self.num_cascade_stages
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeDEQROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        self.box_head = nn.ModuleList()
        self.box_predictor = nn.ModuleList()
        self.box2box_transform = []
        self.proposal_matchers = []
        for k in range(self.num_cascade_stages):
            box_head = cfg.build_box_head(cfg, pooled_shape)
            self.box_head.append(box_head)
            self.box_predictor.append(
                FastRCNNOutputLayers(
                    box_head.output_size, self.num_classes, cls_agnostic_bbox_reg=True
                )
            )
            self.box2box_transform.append(Box2BoxTransform(weights=cascade_bbox_reg_weights[k]))

            if k == 0:
                # The first matching is done by the matcher of ROIHeads (self.proposal_matcher).
                self.proposal_matchers.append(None)
            else:
                self.proposal_matchers.append(
                    Matcher([cascade_ious[k]], [0, 1], allow_low_quality_matches=False)
                )

    def forward(self, images, features, proposals, targets=None, train_step=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        features_list = [features[f] for f in self.in_features]

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features_list, proposals, targets)
            losses.update(self._forward_mask(features_list, proposals, train_step))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances, train_step)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances, train_step=None):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features_list = [features[f] for f in self.in_features]

        instances = self._forward_mask(features_list, instances, train_step)
        instances = self._forward_keypoint(features_list, instances)
        return instances

    def _forward_box(self, features, proposals, targets=None):
        head_outputs = []
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are the input proposals of the next stage
                proposals = self._create_proposals_from_boxes(
                    head_outputs[-1].predict_boxes(), image_sizes
                )
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            head_outputs.append(self._run_stage(features, proposals, k))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, output in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = output.losses()
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h.predict_probs() for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            # Use the boxes of the last head
            boxes = head_outputs[-1].predict_boxes()
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_nms_type,
                self.test_detections_per_img,
            )
            return pred_instances

    def _forward_mask(self, features, instances, train_step):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head(mask_features, train_step)
            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head(mask_features, train_step)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _run_stage(self, features, proposals, stage):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            FastRCNNOutputs: the output of this stage
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor[stage](box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform[stage],
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        return outputs

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals
