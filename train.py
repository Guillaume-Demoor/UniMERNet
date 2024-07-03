import argparse
import time
from tqdm import tqdm
import random
import re
import unimernet.tasks as tasks
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tabulate import tabulate
from rapidfuzz.distance import Levenshtein
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from testunimernet import setup_seeds, load_data, MathDataset, parse_args2
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
from munch import Munch
from tqdm.auto import tqdm
import torch.nn as nn
import pix2tex
from pix2tex.models import get_model
# from pix2tex.utils import *
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check, get_device
from fonction import prepare_batch, padding
from pix2tex.dataset.transforms import train_transform
from pix2tex.eval import evaluate
from functools import partial
import warnings



def main(args):

    setup_seeds()
    # Load Model and Processor
    # cfg = Config(args)
    # vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)

    #print(f'arch_name:{cfg.config.model.arch}')
    #print(f'model_type:{cfg.config.model.model_type}')
    #print(f'checkpoint:{cfg.config.model.finetuned}')
    #print(f'='*100)

    # Generate prediction with MFR model
    #print(f'Device:{device}')
    #print(f'Load model: {end1 - start:.3f}s')

    # transform = transforms.Compose([
    #     vis_processor,
    # ])
    warnings.filterwarnings("ignore", message="The image is already gray")
    fixed_padding = partial(padding, max_width= 672, max_height= 192)

    transform = alb.Compose(
        [
        alb.ToGray(p=1),  # Forcer la conversion en niveaux de gris avec probabilité de 1
        alb.Compose(
            [ 
            alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.15, 0), rotate_limit=1, border_mode=0, interpolation=3, value=[255], p=1),
            alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255], p=0.5)
            ], p=0.15),
        alb.GaussNoise(10, p=0.2),
        alb.RandomBrightnessContrast(0.05, (-0.2, 0), True, p=0.2),
        alb.ImageCompression(95, p=0.3),
        alb.Lambda(image=fixed_padding),
        alb.Normalize((0.7931,), (0.1738,)),  # Adaptation pour un seul canal en niveaux de gris
        ToTensorV2()
        ]
    )
    test_transform = alb.Compose(
        [
        alb.ToGray(p=1),  # Forcer la conversion en niveaux de gris avec probabilité de 1
        alb.Lambda(image=fixed_padding),
        alb.Normalize((0.7931,), (0.1738,)),  # Adaptation pour un seul canal en niveaux de gris
        ToTensorV2()
        ]
    )


    image_path = "/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/data/UniMER-1M/images"
    math_file = "/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/data/UniMER-1M/train.txt"
    start = time.time()
    image_list, math_gts = load_data(image_path, math_file, args)
    end = time.time()
    print("Train data load : ok", " Time : ", end-start, "s")
    dataset = MathDataset(image_list, math_gts, transform=transform)
    dataloader = DataLoader(dataset, batch_size=40, num_workers=14)

    val_im_path = "/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/formulae/val"
    val_math_file = "/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/math.txt"
    val_list, math_val = load_data(val_im_path, val_math_file, args)
    valdataset = MathDataset(val_list, math_val, transform = test_transform)
    valdataloader = DataLoader(valdataset, batch_size=20, num_workers=2)


    print("Dataloader : ok")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = get_device(args, args.no_cuda)
    model = get_model(args)
    if torch.cuda.is_available() and not args.no_cuda:
        gpu_memory_check(model, args)
    max_bleu, max_token_acc = 0, 0
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)


    def save_models(e, step=0):
        torch.save(model.state_dict(), os.path.join(out_path, '%s_e%02d_step%02d.pth' % (args.name, e+1, step)))
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))

    opt = get_optimizer(args.optimizer)(model.parameters(), args.lr, betas=args.betas)
    scheduler = get_scheduler(args.scheduler)(opt, step_size=args.lr_step, gamma=args.gamma)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize
    print("Début de la boucle")
    try:
        k = 0
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            dset = tqdm(iter(dataloader))
            for i, (im, seq) in enumerate(dset):
                im, seq = prepare_batch(im, seq, args)
                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0
                    for j in range(0, len(im), microbatch):
                        tgt_seq, tgt_mask = seq['input_ids'][j:j+microbatch].to(device), seq['attention_mask'][j:j+microbatch].bool().to(device)
                        loss = model.data_parallel(im[j:j+microbatch].to(device), device_ids=args.gpu_devices, tgt_seq=tgt_seq, mask=tgt_mask)*microbatch/dataloader.batch_size
                        loss.backward()  # data parallism loss is a vector
                        total_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    opt.step()
                    scheduler.step()
                    dset.set_description('Loss: %.4f' % total_loss)
                    #if args.wandb:
                        #wandb.log({'train/loss': total_loss})
                else:
                    k +=1
                if ((i+1+len(dataloader)*e) % args.sample_freq == 0) or i == 0:
                    bleu_score, edit_distance, token_accuracy = evaluate(model, valdataloader, args, num_batches=int(args.valbatches*e/args.epochs), name='val')

                    if bleu_score > max_bleu and token_accuracy > max_token_acc:
                        max_bleu, max_token_acc = bleu_score, token_accuracy
                        save_models(e, step=i)
            if (e+1) % args.save_freq == 0:
                save_models(e, step=len(dataloader))
           # if args.wandb:
           #     wandb.log({'train/epoch': e+1})
    except KeyboardInterrupt:
        if e >= 2:
            save_models(e, step=i)
        raise KeyboardInterrupt
    save_models(e, step=len(dataloader))     
    print("Batches refused because of sequence length : ", k, ". Which is : ", k*dataloader.batch_size)

if __name__ == "__main__":
    parsed_args = parse_args2()
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    main(Munch(params))
