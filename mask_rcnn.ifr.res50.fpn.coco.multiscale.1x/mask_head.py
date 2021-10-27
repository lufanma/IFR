# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from cvpods.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from cvpods.modeling.nn_utils import weight_init
from cvpods.utils import get_event_storage

from optimizations import *
from deq2d import *
from mdeq_forward_backward import MDEQWrapper

BN_MOMENTUM = 0.1
DEQ_EXPAND = 1        # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4        # Don't change the value here. The value is controlled by the yaml files.

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU. Corresponds to Figure 2
        in the paper.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, DEQ_EXPAND * planes, stride)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, DEQ_EXPAND * planes, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(DEQ_EXPAND * planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)

        self.downsample = downsample
        self.stride = stride

        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop = VariationalHidDropout2d(dropout)
        if wnorm: self._wnorm()

    def _wnorm(self):
        self.conv1, self.conv1_fn = weight_norm(self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(self.conv2, names=['weight'], dim=0)

    def _reset(self, x=0):
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        # self.drop.reset_mask(x)

    def _copy(self, other):
        self.conv1.weight.data = other.conv1.weight.data.clone()
        self.conv2.weight.data = other.conv2.weight.data.clone()
        # self.drop.mask = other.drop.mask.clone()
        if self.downsample:
            assert False, "Shouldn't be here. Check again"
            self.downsample.weight.data = other.downsample.weight.data
        for i in range(1, 4):
            try:
                eval(f'self.gn{i}').weight.data = eval(f'other.gn{i}').weight.data.clone()
                eval(f'self.gn{i}').bias.data = eval(f'other.gn{i}').bias.data.clone()
            except:
                print(f"Did not set affine=True for gnorm(s) in gn{i}?")

    def forward(self, x, injection=None, *args):
        x = x[0]
        injection = injection[0]

        if injection is None:
            injection = 0

        x = x + injection
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gn3(self.relu3(out))

        return [out]


def mask_rcnn_loss(pred_mask_logits, instances):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )
    return mask_loss


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        self.num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        self.pretrain_steps = cfg.MODEL.ROI_MASK_HEAD.PRETRAINED_STEPS
        self.f_thres = cfg.MODEL.ROI_MASK_HEAD.F_THRES
        self.b_thres = cfg.MODEL.ROI_MASK_HEAD.B_THRES
        # fmt: on

        self.repeat_block = BasicBlock(input_channels, input_channels)

        self.repeat_block_copy = copy.deepcopy(self.repeat_block)
        self.repeat_block._wnorm()

        for params in self.repeat_block_copy.parameters():
            params.requires_grad_(False)

        # self.rcnn_head is basic module
        self.deq = MDEQWrapper(self.repeat_block, self.repeat_block_copy)

        self.deconv = ConvTranspose2d(
            conv_dims if self.num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x, train_step=-1):

        query_feat = [torch.zeros_like(x)]
        x = [x]

        self.repeat_block._reset()
        self.repeat_block_copy._copy(self.repeat_block)

        # feed into deq head
        if 0 <= train_step < self.pretrain_steps:
            for layer_ind in range(self.num_conv):
                query_feat = self.repeat_block(query_feat, x)
        else:
            if train_step == self.pretrain_steps:
                torch.cuda.empty_cache()
            query_feat = self.deq(query_feat,
                                  x,
                                  threshold=self.f_thres,
                                  train_step=train_step,
                                  writer=None)

        x = F.relu(self.deconv(query_feat[0]))
        return self.predictor(x)
