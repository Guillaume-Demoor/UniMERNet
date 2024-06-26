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
from PIL import Image
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
    image_numbers = []
    math_gts = []
    excluded = []
    good_images = []
    # Get the list of image files
    image_names = [f for f in sorted(os.listdir(image_path)) if f.endswith('.png')]
    #all_image_paths = [os.path.join(image_path, f) for f in image_names]
    # Extract image numbers from filenames
    for image_name in tqdm(image_names, desc="Checking images size"):
        try:
            number_str = os.path.splitext(image_name)[0]
            number = int(number_str)
        except ValueError:
            print(f"Filename {image_name} does not match the expected format.")
            continue
        if Image.open(os.path.join(image_path, image_name)) == 'L':
            w, h = Image.open(os.path.join(image_path, image_name)).size
        else:
            w, h = Image.open(os.path.join(image_path, image_name)).convert('L').size
        if args.min_width <= w <= args.max_width and args.min_height <= h <= args.max_height:
            image_numbers.append(number)
            good_images.append(image_name)
        else:
            excluded.append(number) 

    # Read the math file and get the corresponding labels
    with open(math_file, 'r') as f:
        lines = f.readlines()
        for number in image_numbers:
            if number < len(lines):
                math_gts.append(lines[number].strip())
            else:
                print(f"No corresponding line for image number {number}")
    good_image_paths = [os.path.join(image_path, f) for f in good_images]
    print("Images rejected because of size : ", len(excluded), "/", len(image_names), flush=True)
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
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
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


def main():

    setup_seeds()
    # Load Model and Processor
    start = time.time()
    cfg = Config(parse_args())
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    model.to(device)

    print(f'arch_name:{cfg.config.model.arch}')
    print(f'model_type:{cfg.config.model.model_type}')
    print(f'checkpoint:{cfg.config.model.finetuned}')
    print(f'='*100)

    end1 = time.time()

    # Generate prediction with MFR model
    print(f'Device:{device}')
    print(f'Load model: {end1 - start:.3f}s')

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



    for val_name, image_path, math_file in zip(val_names, image_paths, math_files):
        image_list, math_gts = load_data(image_path, math_file)

        transform = transforms.Compose([
            vis_processor,
        ])

        dataset = MathDataset(image_list, math_gts, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=16)
        
        math_preds = []
        for images, label in tqdm(dataloader):
            images = images.to(device)
            with torch.no_grad():
                output = model.generate({"image": images})
            math_preds.extend(output["pred_str"])

        # Compute BLEU/METEOR/EditDistance
        norm_gts = [normalize_text(gt) for gt in math_gts]
        norm_preds = [normalize_text(pred) for pred in math_preds] 
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
        print(f'Inference Time: {end2 - end1}s')
        print(tabulate(score_table, headers=[*score_headers]))
        print('='*100)

if __name__ == "__main__":
    main()
