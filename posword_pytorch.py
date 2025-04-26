
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# RNN Module for word or POS
class RNNForLM(nn.Module):
    def __init__(self, n_vocab, n_units):
        super(RNNForLM, self).__init__()
        self.embed = nn.Embedding(n_vocab, n_units)
        self.lstm1 = nn.LSTM(n_units, n_units, batch_first=True)
        self.lstm2 = nn.LSTM(n_units, n_units, batch_first=True)
        self.linear = nn.Linear(n_units, n_vocab)

    def forward(self, x):
        h = self.embed(x)
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        h = self.linear(h)
        return h

# Final model combining word and pos RNNs
class Rec2Network(nn.Module):
    def __init__(self, n_vocab, n_word_units, n_pos, n_pos_units):
        super(Rec2Network, self).__init__()
        n_lin_units = n_vocab + n_pos
        self.wordlm = RNNForLM(n_vocab, n_word_units)
        self.poslm = RNNForLM(n_pos, n_pos_units)
        self.lin1 = nn.Linear(n_lin_units, n_lin_units)
        self.lin2 = nn.Linear(n_lin_units, n_lin_units)
        self.soft = nn.Linear(n_lin_units, n_vocab)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        y1 = torch.softmax(self.wordlm(x1), dim=-1)
        y2 = torch.softmax(self.poslm(x2), dim=-1)

        y = torch.cat((y1, y2), dim=-1)

        h = self.lin1(y)
        h = self.lin2(h)
        out = self.soft(h)

        return out

# Dataset to manage word/POS pairs
class ParallelSequentialDataset(Dataset):
    def __init__(self, data, bprop_len):
        self.data = data
        self.bprop_len = bprop_len

    def __len__(self):
        return len(self.data) - self.bprop_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.bprop_len]
        y = self.data[idx+1:idx+self.bprop_len+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation loop
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20)
    parser.add_argument('--vocabulary', '-V', type=int, default=7900)
    parser.add_argument('--bproplen', '-l', type=int, default=35)
    parser.add_argument('--epoch', '-e', type=int, default=39)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--r1units', '-r1', type=int, default=100)
    parser.add_argument('--r2units', '-r2', type=int, default=50)
    parser.add_argument('--dunits', '-d', type=int, default=150)
    parser.add_argument('--r1layers', '-r1l', type=int, default=1)
    parser.add_argument('--r2layers', '-r2l', type=int, default=1)
    parser.add_argument('--model', '-m', default='model.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    # Fake dataset, replace this with real one
    dataset = np.random.randint(0, args.vocabulary, size=(50000, 2))
    train_data = dataset[:int(0.7 * len(dataset))]
    val_data = dataset[int(0.7 * len(dataset)):int(0.9 * len(dataset))]

    train_dataset = ParallelSequentialDataset(train_data, args.bproplen)
    val_dataset = ParallelSequentialDataset(val_data, args.bproplen)

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize)

    n_vocab = np.max(dataset[:, 0]) + 1
    n_pos = np.max(dataset[:, 1]) + 1

    model = Rec2Network(n_vocab, args.r1units, n_pos, args.r2units).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())

    for epoch in range(1, args.epoch + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {math.exp(val_loss):.2f}')

    # Save model
    torch.save(model.state_dict(), args.model)

if __name__ == '__main__':
    main()
