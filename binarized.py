import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from rep import BinaryOutputToken, BinaryOutputRep
from model import BasicModel, BinarizedModel
from dataset import TokenToExpressionDataset, collate_fn
from rich.progress import track
from collections import deque
from main import compute_accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TokenToExpressionDataset(N=int(1e4))
    test_dl = DataLoader(
        test_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
    )

    model = BinarizedModel(BinaryOutputToken, BinaryOutputRep)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    opt = Adam(model.parameters(), lr=1e-4)

    acc = compute_accuracy(model, test_dl)
    print(acc)

    for epoch in range(20):

        dataset = TokenToExpressionDataset(N=int(1e5))
        train_dl = DataLoader(
            dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
        )

        losses = deque(maxlen=100)

        N = len(train_dl)
        for (i, (x, y)) in enumerate(train_dl):
            x = x.to(device)
            y = y.to(device)
            out = model(x, y)

            y = y[:, 1:].flatten()
            out = out[:, :-1, :].reshape(-1, BinaryOutputToken.size())

            out = out[y != -1]
            y = y[y != -1]

            opt.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            losses.append(loss.item())

            if i % 20 == 0 and i > 0:
                print(f"({epoch} {i}/{N}) Loss: {np.mean(losses)}")

        acc = compute_accuracy(model, test_dl)
        print(f"({epoch}) Loss: {np.mean(losses)} Accuracy: {acc}")
