import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ZOO = {
    "resnet50":     (models.resnet50,         models.ResNet50_Weights.IMAGENET1K_V1),
    "vgg16":        (models.vgg16,            models.VGG16_Weights.IMAGENET1K_V1),
    "alexnet":      (models.alexnet,          models.AlexNet_Weights.IMAGENET1K_V1),
    "resnet101":    (models.resnet101,        models.ResNet101_Weights.IMAGENET1K_V1),
    "mobilenet":    (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
    "efficientnet": (models.efficientnet_b0,  models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}

def load_pretrained(name):
    ctor, weights = MODEL_ZOO[name]
    model = ctor(weights=weights).to(DEVICE).eval()
    preprocess = weights.transforms()        
    categories = weights.meta["categories"]
    return model, preprocess, categories


def load_and_preprocess_image(img_path, preprocess):
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img)                  
    return tensor.unsqueeze(0).to(DEVICE)    


@torch.no_grad()
def make_predictions(model, img_tensor, categories, top_k=5):
    logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(top_k)
    return [(categories[i], p.item()) for i, p in zip(top_idx.tolist(), top_probs)]


def print_top_5_predictions(name, img_path):
    model, preprocess, categories = load_pretrained(name)
    img_tensor = load_and_preprocess_image(img_path, preprocess)

    start = time.time()
    predictions = make_predictions(model, img_tensor, categories)
    elapsed = (time.time() - start) * 1000

    print(f"\nTop 5 predictions from {name.upper()} ({elapsed:.0f} ms):")
    for i, (label, prob) in enumerate(predictions, 1):
        print(f"{i}. {label} ({prob * 100:.2f}% probability)")

    # free memory before loading the next model
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_predict(args):
    for name in args.models:
        print_top_5_predictions(name, args.image)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    all_models = list(MODEL_ZOO.keys())

    p = sub.add_parser("predict", help="Top-5 ImageNet predictions for one image")
    p.add_argument("--image", required=True, help="path to your image (use your own path)")
    p.add_argument("--models", nargs="+", default=all_models, choices=all_models)
    p.set_defaults(func=run_predict)

    args = parser.parse_args()
    print(f"Using device: {DEVICE}")
    args.func(args)

if __name__ == "__main__":
    main()