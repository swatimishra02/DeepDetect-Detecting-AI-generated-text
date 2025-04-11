# import torch
# import numpy as np
# import os
# from transformers import XLNetTokenizer, XLNetModel
# from model import FeatureClassifier
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Lazy global instances
# _tokenizer = None
# _xlnet_model = None
# _classifier = None

# def get_tokenizer():
#     global _tokenizer
#     if _tokenizer is None:
#         _tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
#     return _tokenizer

# def get_xlnet_model():
#     global _xlnet_model
#     if _xlnet_model is None:
#         _xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased').to(device)
#         _xlnet_model.eval()
#     return _xlnet_model

# def get_classifier_model():
#     global _classifier
#     if _classifier is None:
#         INPUT_DIM = 770
#         _classifier = FeatureClassifier(input_dim=INPUT_DIM)
#         _classifier.load_state_dict(torch.load("best_model.pth", map_location=device))
#         _classifier.to(device)
#         _classifier.eval()
#     return _classifier

# def extract_features_from_text(text: str) -> np.ndarray:
#     tokenizer = get_tokenizer()
#     xlnet_model = get_xlnet_model()

#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = xlnet_model(**inputs)
#         xlnet_features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

#     extra_features = np.array([0.0, 0.0])
#     combined_features = np.concatenate((xlnet_features, extra_features))

#     return combined_features

# def predict(features: np.ndarray) -> tuple[str, float]:
#     model = get_classifier_model()
#     tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

#     with torch.no_grad():
#         logits = model(tensor)
#         probs = F.softmax(logits, dim=1).cpu().numpy()[0]
#         pred_idx = int(np.argmax(probs))
#         confidence = float(probs[pred_idx]) * 100  # convert to percentage

#     label = "AI-generated" if pred_idx == 1 else "Human-written"
#     return label, round(confidence, 2)

import torch
import numpy as np
import os
import gc
import torch.nn.functional as F
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

    # Additional dummy features (e.g. for structure similarity, grammar etc.)
    extra_features = np.array([0.0, 0.0])
    combined_features = np.concatenate((xlnet_features, extra_features))

    # Cleanup
    del inputs, outputs, xlnet_features
    torch.cuda.empty_cache()
    gc.collect()

    return combined_features

def predict(features: np.ndarray) -> tuple[str, float]:
    model = get_classifier_model()
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100  # Convert to %

    label = "AI-generated" if pred_idx == 1 else "Human-written"

    # Cleanup
    del tensor, logits, probs
    torch.cuda.empty_cache()
    gc.collect()

    return label, round(confidence, 2)

