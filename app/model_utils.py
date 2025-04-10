import torch
import numpy as np
from model import FeatureClassifier 
import os
from transformers import XLNetTokenizer, XLNetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased').to(device)

def extract_features_from_text(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = xlnet_model(**inputs)
        xlnet_features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    extra_features = np.array([0.0, 0.0])  
    combined_features = np.concatenate((xlnet_features, extra_features))

    return combined_features


INPUT_DIM = 770  
model = FeatureClassifier(input_dim=INPUT_DIM)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

def predict(features: np.ndarray) -> str:
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return "AI-generated" if pred == 1 else "Human-written"
