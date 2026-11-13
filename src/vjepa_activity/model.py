"""Load the V-JEPA2 backbone and notebook-trained classifier."""

from pathlib import Path
import torch
from transformers import AutoModel, AutoVideoProcessor
from .classifier import PooledMLPClassifier


def load_pipeline(checkpoint_path: str, backbone_name: str, device_name: str = "auto"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {checkpoint_path}. See README.md; the notebook currently saves features, not classifier weights."
        )
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes")
    if not classes or len(classes) != 101:
        raise ValueError("Checkpoint must contain the ordered 'classes' list for all 101 UCF-101 labels.")
    config = checkpoint.get("model_config", {})
    classifier = PooledMLPClassifier(
        embed_dim=int(config.get("embed_dim", 1024)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_classes=len(classes),
        dropout=float(config.get("dropout", 0.3)),
    )
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(device).eval()
    backbone = AutoModel.from_pretrained(backbone_name).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(backbone_name)
    return backbone, processor, classifier, classes, device
