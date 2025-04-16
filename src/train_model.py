import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os

class EssayDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_len):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = self.scores[idx]
        tokens = self.tokenizer(text)
        tokens = tokens[:self.max_len]
        padding_length = self.max_len - len(tokens)
        tokens += [0] * padding_length

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'targets': torch.tensor(score, dtype=torch.float)
        }

def simple_tokenizer(text):
    return [ord(c) for c in text.lower() if ord(c) < 128]  # simple ASCII encoding

class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64):
        super(BiLSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        output = lstm_out[:, -1, :]
        return torch.sigmoid(self.fc(output).squeeze()) * 10  # Output range 0–10

def create_data_loaders(df, tokenizer, max_len, batch_size):
    dataset = EssayDataset(
        texts=df['essay'].to_numpy(),
        scores=df['score'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        outputs = model(input_ids)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids)
            pred = outputs.cpu().detach().numpy()
            if pred.ndim == 0:
                predictions.append(pred.item())
            else:
                predictions.extend(pred.tolist())

            targ = targets.cpu().numpy()
            if targ.ndim == 0:
                actuals.append(targ.item())
            else:
                actuals.extend(targ.tolist())

    mse = mean_squared_error(actuals, predictions)
    return mse, predictions, actuals

def run_training(data_path='data/processed/essays.csv', model_path='models/bilstm_essay_model.pt'):
    df = pd.read_csv(data_path)
    df = df[['essay', 'score']].dropna()
    df['score'] = df['score'].astype(float)
    print(f"✅ Loaded {len(df)} essays from {data_path}")

    MAX_LEN = 300
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3
    VOCAB_SIZE = 128  # ASCII characters only

    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    train_loader = create_data_loaders(df_train, simple_tokenizer, MAX_LEN, BATCH_SIZE)
    val_loader = create_data_loaders(df_val, simple_tokenizer, MAX_LEN, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMRegressor(vocab_size=VOCAB_SIZE)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_mse, _, _ = eval_model(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == '__main__':
    run_training()
