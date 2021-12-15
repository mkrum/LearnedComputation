import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import numpy as np

from rich.progress import track
from collections import deque

from lc.rep import ExpressionRep
from lc.model import BasicModel
from lc.dataset import MathDataset, TestMathDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pred(test_string, model):

    act = nn.LogSoftmax(dim=-1)
    tokens = ["<start>"]

    state = model.input_expression.from_str_list(test_string).to_tensor()
    state = state.unsqueeze(0).cuda()

    next_token = ""
    while next_token != "<stop>":
        out = torch.tensor(
            [
                model.output_token.from_str(tokens[i]).to_int()
                for i in range(len(tokens))
            ]
        )
        out = out.unsqueeze(0).cuda()

        with torch.no_grad():
            logits = model(state, out)[:, -1, :]

        dist = Categorical(logits=logits)
        next_token = model.output_token.from_int(dist.sample().cpu().item()).to_str()
        tokens.append(next_token)

    return "".join(tokens[1:-1])


def compute_accuracy(model, test_dl):

    N = 10

    x, y, x_raw, y_raw = next(iter(test_dl))

    equations = x_raw[:N]
    targets = y_raw[:N]

    out = model.inference(x[:N].cuda())

    for i in range(N):
        print(f'{" ".join(equations[i])} {targets[i]} {out[i]}')

    valid = 0.0
    error = 0.0
    correct = 0.0
    total = 0.0

    for (x, y, _, targets) in test_dl:
        x = x.to(device)

        out = model.inference(x)

        for (o, t) in zip(out, targets):
            t = int(t)
            if o is not None:
                valid += 1
                error += (o - t) ** 2

            if o == t:
                correct += 1

        total += x.shape[0]

    print(error / valid)
    print(valid / total)
    print(correct / total)
    return correct / total


def train(
    model_type,
    input_expression,
    output_expression,
    lr=1e-4,
    num_range=(0, 127),
    batch_size=256,
    train_examples=int(1e5),
):

    test_dataset = TestMathDataset(
        input_expression,
        output_expression,
        N=int(1e4),
        num_range=num_range,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=TestMathDataset.collate_fn,
    )

    model = model_type(input_expression, output_expression)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    opt = Adam(model.parameters(), lr=lr)

    acc = compute_accuracy(model, test_dl)
    print(acc)

    for epoch in range(20):

        dataset = MathDataset(
            input_expression, output_expression, N=train_examples, num_range=num_range
        )
        train_dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MathDataset.collate_fn,
        )

        losses = deque(maxlen=100)

        N = len(train_dl)
        for (i, (x, y)) in enumerate(train_dl):
            x = x.to(device)
            y = y.to(device)
            out = model(x, y)

            y = y[:, 1:].flatten()
            out = out[:, :-1, :].reshape(-1, model.output_token.size())

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


if __name__ == "__main__":
    train(BasicModel, ExpressionRep, ExpressionRep)
