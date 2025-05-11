
import torch
import torch.nn as nn
from src.train_model import BiLSTMRegressor, simple_tokenizer

class EssayScorer:
    def __init__(self, model_path='models/bilstm_essay_model.pt'):
        self.model = BiLSTMRegressor(vocab_size=128)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, essay):
        tokens = simple_tokenizer(essay)
        tokens = tokens[:300]
        tokens += [0] * (300 - len(tokens))
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            output = self.model(input_tensor).item()
        return round(output, 2)
