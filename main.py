from dataset import SimpleExpressionDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
import torch
import torch.nn as nn
from torch.optim import Adam
import math

from rep import MathToken


def collate_fn(batch):
    data, labels = zip(*batch)
    padded_data = pad_sequence(data, batch_first=True, padding_value=-1)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return padded_data, padded_labels


dataset = SimpleExpressionDataset(N=int(1e5))
train_dl = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)


def get_mask(data):
    return data != -1


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(MathToken.size(), 128)
        self.transformer = nn.Transformer(
            num_encoder_layers=6, num_decoder_layers=6, batch_first=True, d_model=128
        )
        self.pe = positionalencoding1d(128, 1000).cuda()
        self.to_dist = nn.Sequential(nn.Linear(128, MathToken.size()))

    def forward(self, data, output):
        mask = get_mask(data).cuda()
        tgt_mask = get_mask(output).cuda()

        embedded_data = self.embed(mask * data) + self.pe[: data.shape[1]].unsqueeze(0)
        embedded_tgt = self.embed(tgt_mask * output) + self.pe[
            : output.shape[1]
        ].unsqueeze(0)
        attn_mask = self.transformer.generate_square_subsequent_mask(
            output.shape[1]
        ).cuda()
        out = self.transformer(
            embedded_data,
            embedded_tgt,
            src_key_padding_mask=~mask,
            tgt_key_padding_mask=~tgt_mask,
            tgt_mask=attn_mask,
        )
        return self.to_dist(out)


model = BasicModel()
model.cuda()

loss_fn = nn.CrossEntropyLoss()

opt = Adam(model.parameters(), lr=1e-4)
for _ in range(5):
    for (x, y) in train_dl:
        x = x.cuda()
        y = y.cuda()
        out = model(x, y)

        y = y[:, 1:].flatten()
        out = out[:, :-1, :].reshape(-1, MathToken.size())

        out = out[y != -1]
        y = y[y != -1]

        opt.zero_grad()
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        print(loss.item())


def print_pred(test_string):
    test_string = list(test_string)
    state = ExpressionRep.from_str_list(test_string).to_tensor().unsqueeze(0).cuda()
    act = nn.LogSoftmax(dim=-1)
    tokens = ["<start>"]

    next_token = ""
    while next_token != "<stop>":
        out = ExpressionRep.from_str_list(tokens).to_tensor().unsqueeze(0).cuda()
        with torch.no_grad():
            logits = model(state, out)[:, -1, :]
        dist = Categorical(logits=logits)
        next_token = MathToken.from_int(dist.sample().cpu().item()).to_str()
        tokens.append(next_token)

    print("".join(tokens[1:-1]))
