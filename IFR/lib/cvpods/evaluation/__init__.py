# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .build import build_evaluator
from .coco_evaluation import CustomizedCOCOEvaluator
from .evaluator import (
    BaseDatasetEvaluator,
    BaseDatasetEvaluators,
    inference_context,
    inference_on_dataset,
    inference_on_files
)
from .registry import EVALUATOR
from .testing import print_csv_format, verify_results

__all__ = [k for k in globals().keys() if not k.startswith("_")]