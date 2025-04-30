# train.py
import argparse, math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model.rec2_model import Rec2Network
from utils import load_brown_dataset, build_vocab, encode

class ParallelSequentialDataset(Dataset):
    def __init__(self, data, bprop_len):
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.bprop_len = bprop_len
        self.length = len(self.x) // bprop_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_seq = self.x[idx * self.bprop_len : (idx + 1) * self.bprop_len]
        y_seq = self.y[idx * self.bprop_len : (idx + 1) * self.bprop_len]
        return torch.LongTensor(x_seq), torch.LongTensor(y_seq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1units', type=int, default=64)
    parser.add_argument('--r2units', type=int, default=64)
    parser.add_argument('--bproplen', type=int, default=32)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = load_brown_dataset()
    word_vocab, pos_vocab = build_vocab(pairs)
    data = encode(pairs, word_vocab, pos_vocab)

    n_vocab = len(word_vocab)
    n_pos = len(pos_vocab)

    # Split
    train_data = data[:int(0.7 * len(data))]
    val_data = data[int(0.7 * len(data)):int(0.9 * len(data))]

    train_dataset = ParallelSequentialDataset(train_data, args.bproplen)
    val_dataset = ParallelSequentialDataset(val_data, args.bproplen)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize)

    model = Rec2Network(n_vocab, args.r1units, n_pos, args.r2units).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch, y_batch[:, :-1])
            loss = loss_fn(output.reshape(-1, output.size(-1)), y_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch, y_batch[:, :-1])
                loss = loss_fn(output.reshape(-1, output.size(-1)), y_batch[:, 1:].reshape(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Perplexity: {math.exp(val_loss):.2f}")

if __name__ == "__main__":
    main()
