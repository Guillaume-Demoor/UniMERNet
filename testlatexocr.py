from pix2tex.eval import evaluate 
from testunimernet import parse_args2
import torch
import argparse
from pix2tex.utils import get_device
from pix2tex.models import get_model
import warnings 
import albumentations as alb
from functools import partial
from fonction import padding 
from albumentations.pytorch import ToTensorV2
from testunimernet import MathDataset, DataLoader, load_data
import yaml
from munch import Munch

def main(args):
    # Modèle LaTeX-OCR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = get_device(args, args.no_cuda)
    print(f'Device:{device}')
    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint, device))
    #model.to(device)

    # tokenizer = PreTrainedTokenizerFast(tokenizer_file= args.tokenizer)
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    image_path = "/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/formulae/test"
    math_file = "/home/gdemoor/warm/TestGuill/UniMERNet/UniMERNet/math.txt"
    image_list, math_gts = load_data(image_path, math_file, args)
    dataset = MathDataset(image_list, math_gts, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=10, num_workers=16)
    bleu, edit, acc = evaluate(model, dataloader, args, None, 'Evaluation')
    print(f"BLEU Score: {bleu:.2f}, Edit Distance: {edit:.2f}, Accuracy: {acc:.2f}")

if __name__ == "__main__":
    parsed_args = parse_args2()
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    main(Munch(params))