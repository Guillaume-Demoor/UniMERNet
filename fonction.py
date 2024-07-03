import argparse
import time
from tqdm import tqdm
import evaluate
import random
import re
import unimernet.tasks as tasks

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tabulate import tabulate
from rapidfuzz.distance import Levenshtein
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unimernet.common.config import Config
from unimernet.datasets.builders import *
from unimernet.models import *
from unimernet.processors import *
from unimernet.tasks import *
from unimernet.processors import load_processor
import os
import argparse
import logging
import yaml
import cv2
from transformers import PreTrainedTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from munch import Munch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from pix2tex.models import get_model
# from pix2tex.utils import *
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check
from pix2tex.dataset.transforms import train_transform

def prepare_batch(ims, eqs, args):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    tok = tokenizer(list(eqs), return_token_type_ids=False)
    # pad with bos and eos token
    for k, p in zip(tok, [[args.bos_token_id, args.eos_token_id], [1, 1]]):
        tok[k] = pad_sequence([torch.LongTensor([p[0]]+x+[p[1]]) for x in tok[k]], batch_first=True, padding_value=args.pad_token_id)
    # check if sequence length is too long
    if args.max_seq_len < tok['attention_mask'].shape[1]:
        return None, None
    try:
        ims = torch.cat(list(ims)).float().unsqueeze(1)
    except RuntimeError:
        logging.critical('Images not working: %s' % (' '.join(list(ims))))
        return None, None
    return ims, tok

def padding(image, max_width, max_height, **kwargs):
        # Calculer le padding nécessaire pour chaque côté
        height, width = image.shape[:2]
        top = (max_height - height) // 2
        bottom = max_height - height - top
        left = (max_width - width) // 2
        right = max_width - width - left
        # Appliquer le padding
        padded_image = np.pad(image, ((top, bottom), (left, right)), 'constant', constant_values=255)
        return padded_image