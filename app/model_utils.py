import torch
import numpy as np
import os
from transformers import XLNetTokenizer, XLNetModel
from model import FeatureClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy global instances
_tokenizer = None
_xlnet_model = None
_classifier = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    return _tokenizer

def get_xlnet_model():
    global _xlnet_model
    if _xlnet_model is None:
        _xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased').to(device)
        _xlnet_model.eval()
    return _xlnet_model

def get_classifier_model():
    global _classifier
    if _classifier is None:
        INPUT_DIM = 770
        _classifier = FeatureClassifier(input_dim=INPUT_DIM)
        _classifier.load_state_dict(torch.load("best_model.pth", map_location=device))
        _classifier.to(device)
        _classifier.eval()
    return _classifier

def extract_features_from_text(text: str) -> np.ndarray:
    tokenizer = get_tokenizer()
    xlnet_model = get_xlnet_model()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = xlnet_model(**inputs)
        xlnet_features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    extra_features = np.array([0.0, 0.0])
    combined_features = np.concatenate((xlnet_features, extra_features))

    return combined_features

def predict(features: np.ndarray) -> str:
    model = get_classifier_model()
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return "AI-generated" if pred == 1 else "Human-written"
