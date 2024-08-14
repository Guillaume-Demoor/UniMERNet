import argparse
import time
from tqdm import tqdm
import evaluate
import random
import re
import unimernet.tasks as tasks
import torch.nn.functional as F
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, UnidentifiedImageError
from tabulate import tabulate
from rapidfuzz.distance import Levenshtein
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unimernet.common.config import Config
import pix2tex 
# imports modules for registration
from unimernet.datasets.builders import *
from unimernet.models import *
from unimernet.processors import *
from unimernet.tasks import *
from unimernet.processors import load_processor
from pix2tex.cli import LatexOCR
from pix2tex.models import get_model
import yaml
from transformers import AutoTokenizer, AutoModel
from munch import Munch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from fonction import padding, prepare_batch
from functools import partial
import warnings
from pix2tex.utils import get_device, seed_everything
from pix2tex.eval import detokenize
from transformers import PreTrainedTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from transformers import TrOCRProcessor

class MathDataset(Dataset):
    def __init__(self, image_paths, math_gts, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.math_gts = math_gts

    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if image.mode != 'L':
            image = image.convert('L')
        label = self.math_gts[idx]
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, label

def load_data(image_path, math_file, args):
    math_gts = []
    excluded_im = []
    excluded_seq = []
    good_images = []
    # Get the list of image files
    image_names = [f for f in sorted(os.listdir(image_path)) if f.endswith('.png')]
    # Get labels from txt file
    with open(math_file, 'r') as f:
        lines = f.readlines()

    for image_name in tqdm(image_names, desc="Checking images size"):
        try:
            number_str = os.path.splitext(image_name)[0]
            number = int(number_str)
        except ValueError:
            print(f"Filename {image_name} does not match the expected format.")
            continue
        try:
            if Image.open(os.path.join(image_path, image_name)) == 'L':
                w, h = Image.open(os.path.join(image_path, image_name)).size
            else:
                w, h = Image.open(os.path.join(image_path, image_name)).convert('L').size
        except UnidentifiedImageError:
            continue 
        if args.min_width <= w <= args.max_width and args.min_height <= h <= args.max_height:
            if number < len(lines):
                line = lines[number].strip()
                if len(line) < args.max_seq_len:
                    good_images.append(image_name)
                    math_gts.append(line)
                else:
                    excluded_seq.append(line)
            else:
                print(f"No corresponding line for image number {number}")
        else:
            excluded_im.append(number) 

    good_image_paths = [os.path.join(image_path, f) for f in good_images]
    print("Images rejected because of size : ", len(excluded_im), "/", len(image_names), flush=True)
    print("Images rejected because of sequence size : ", len(excluded_seq), "/", len(image_names), flush=True)
    print(len(good_image_paths), len(math_gts))
    return good_image_paths, math_gts


def normalize_text(text):
    """Remove unnecessary whitespace from LaTeX code."""
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda match: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', text)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == text:
            break
    return text


def score_text(predictions, references):
    bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1,1e8))
    bleu_results = bleu.compute(predictions=predictions, references=references)

    lev_dist = []
    for p, r in zip(predictions, references):
        lev_dist.append(Levenshtein.normalized_distance(p, r))

    return {
        'bleu': bleu_results["bleu"],
        'edit': sum(lev_dist) / len(lev_dist)
    }


def setup_seeds(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_args2():
    """ Parse les arguments de la ligne de commande. """
    parser = argparse.ArgumentParser(description="Lire un fichier de configuration YAML.")
    parser.add_argument("--config", type=str, help="Chemin vers le fichier de configuration YAML.")
    args = parser.parse_args()
    return args



    
def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", required=True, help="path to configuration file.")
    parser.add_argument("--result_path", type=str, help="Path to json file to save result to.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def main(args):

    seed_everything(args.seed)
    
    # Modèle UniMERnet
    # cfg = Config(parse_args())
    # task = tasks.setup_task(cfg)
    # #model = task.build_model(cfg)
    # model = AutoModel.from_pretrained("wanderkid/unimernet")


    # Modèle LaTeX-OCR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = get_device(args, args.no_cuda)
    model = get_model(args)
    print(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, device))
    # model.to(device)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file= args.tokenizer)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f'Device:{device}')

    # Load Data (image and corresponding annotations)
    val_names = [
        "Simple Print Expression(SPE)",
        "Complex Print Expression(CPE)",
        "Screen Capture Expression(SCE)",
        "Handwritten Expression(HWE)"
    ]
    image_paths = [
        "./data/UniMER-Test/spe",
        "./data/UniMER-Test/cpe",
        "./data/UniMER-Test/sce",
        "./data/UniMER-Test/hwe"
    ]
    math_files = [
        "./data/UniMER-Test/spe.txt",
        "./data/UniMER-Test/cpe.txt",
        "./data/UniMER-Test/sce.txt",
        "./data/UniMER-Test/hwe.txt"
    ]
    # val_names = ["LaTeX-OCR"]
    # image_paths = ["/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/formulae/test"]
    # math_files  = ["/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/math.txt"]


    for val_name, image_path, math_file in zip(val_names, image_paths, math_files):
        image_list, math_gts = load_data(image_path, math_file, args)


        warnings.filterwarnings("ignore", message="The image is already gray")
        fixed_padding = partial(padding, max_width= 672, max_height= 192)
        test_transform = alb.Compose(
        [
        alb.ToGray(p=1),  # Forcer la conversion en niveaux de gris avec probabilité de 1
        alb.Lambda(image=fixed_padding),
        alb.Normalize((0.7931,), (0.1738,)),  # Adaptation pour un seul canal en niveaux de gris
        ToTensorV2()
        ]
        )

        dataset = MathDataset(image_list, math_gts, transform=test_transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=16)
        
        norm_preds = []
        norm_gts = []
        for images, labels in tqdm(dataloader):
            images, labels = prepare_batch(images, labels, args)
            if labels is not None and images is not None:
                images = images.to(device)
                with torch.no_grad():
                    output = model.generate(images, temperature=0.2)
                output = detokenize(output, tokenizer)
                output = [''.join(sublist) for sublist in output]
                labels = detokenize(labels['input_ids'], tokenizer)
                labels = [''.join(sublist) for sublist in labels]
                norm_preds.extend([normalize_text(seq) for seq in output])
                norm_gts.extend([normalize_text(label) for label in labels])

        print(f'len_gts:{len(norm_gts)}, len_preds={len(norm_preds)}')
        print(f'norm_gts[0]:{norm_gts[0]}')
        print(f'norm_preds[0]:{norm_preds[0]}')

        p_scores = score_text(norm_preds, norm_gts)

        write_data= {
            "scores": p_scores,
            "text": [{"prediction": p, "reference": r} for p, r in zip(norm_preds, norm_gts)]
        }

        score_table = []
        score_headers = ["bleu", "edit"]
        score_dirs = ["⬆", "⬇"]

        score_table.append([write_data["scores"][h] for h in score_headers])

        score_headers = [f"{h} {d}" for h, d in zip(score_headers, score_dirs)]

        end2 = time.time()

        print(f'Evaluation Set:{val_name}')
        print(tabulate(score_table, headers=[*score_headers]))
        print('='*100)

if __name__ == "__main__":
    parsed_args = parse_args2()
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    main(Munch(params))
