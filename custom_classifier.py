import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
import numpy as np
import gensim.downloader as api  # For loading pre-trained embeddings

# === 1. Load Pre-trained Embeddings (GloVe) === #
def load_glove_embeddings(embedding_name='glove-wiki-gigaword-100'):
    glove = api.load(embedding_name)
    return glove

# === 2. Tokenizer & Vocabulary === #
class TextDataset(Dataset):
    def __init__(self, texts, labels, glove_model):
        self.labels = labels
        self.glove = glove_model
        self.dim = glove_model.vector_size
        self.texts = [self.text_to_tensor(text) for text in texts]

    def text_to_tensor(self, text):
        words = text.lower().split()
        vectors = [self.glove[word] for word in words if word in self.glove]
        if not vectors:
            return torch.zeros(self.dim)
        return torch.tensor(np.mean(vectors, axis=0), dtype=torch.float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# === 3. Classifier Architecture === #
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# === 4. Training Loop === #
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# === 5. Evaluation === #
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# === 6. Example Usage === #
if __name__ == "__main__":
    # Example data
    texts = ["This is a positive sentence", "I hate this thing", "What a great day!", "Terrible experience"]
    labels = torch.tensor([1, 0, 1, 0])  # 1: Positive, 0: Negative

    glove = load_glove_embeddings()  # Load GloVe vectors
    dataset = TextDataset(texts, labels, glove)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    input_dim = glove.vector_size
    model = TextClassifier(input_dim, hidden_dim=64, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(5):
        loss = train_model(model, dataloader, optimizer, criterion)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    # Evaluation
    metrics = evaluate_model(model, dataloader)
    print("Evaluation metrics:", metrics)
