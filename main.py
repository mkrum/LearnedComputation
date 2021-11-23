import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from rep import MathToken
from model import BasicModel
from dataset import SimpleExpressionDataset, collate_fn
from rich.progress import track
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_accuracy(test_dl):

    correct = 0.0
    total = 0.0

    for (x, y) in track(test_dl):
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)

        y = y[:, 1:].flatten()
        out = out[:, :-1, :].reshape(-1, MathToken.size())
        out = out[y != -1]
        y = y[y != -1]
        preds = torch.argmax(out, axis=-1)
        correct += torch.sum(preds == y).item()
        total += preds.shape[0]

    return correct / total


test_dataset = SimpleExpressionDataset(N=int(1e4))
test_dl = DataLoader(test_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

model = BasicModel()
model.to(device)

loss_fn = nn.CrossEntropyLoss()

opt = Adam(model.parameters(), lr=1e-4)

acc = compute_accuracy(test_dl)
print(acc)

for epoch in range(5):

    dataset = SimpleExpressionDataset(N=int(1e5))
    train_dl = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

    losses = deque(maxlen=100)

    N = len(train_dl)
    for (i, (x, y)) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)

        y = y[:, 1:].flatten()
        out = out[:, :-1, :].reshape(-1, MathToken.size())

        out = out[y != -1]
        y = y[y != -1]

        opt.zero_grad()
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if i % 100 == 0 and i > 0:
            print(f"({epoch} {i}/{N}) Loss: {np.mean(losses)}")

    acc = compute_accuracy(test_dl)
    print(f"({epoch}) Loss: {np.mean(losses)} Accuracy: {acc}")
