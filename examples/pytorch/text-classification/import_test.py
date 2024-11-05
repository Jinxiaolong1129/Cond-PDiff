import sys
# sys.path.append("/home/jin509/p-diff/lora-diff/src/transformers")  # Add the path to Python's search path
sys.path.append("./src/transformers")  # Add the path to Python's search path

import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from transformers import (
    AdapterConfig,
    LoRAConfig,
    AdapterTrainer,
    AutoAdapterModel,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import transformers.adapters

import transformers.adapters.composition as ac
