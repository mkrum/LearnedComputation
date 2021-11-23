import math
import torch
import torch.nn as nn
from rep import MathToken
from torch.distributions import Categorical


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
        self.pe = positionalencoding1d(128, 1000)
        self.to_dist = nn.Sequential(nn.Linear(128, MathToken.size()))

    def _encode_position(self, data):

        if self.pe.device != data.device:
            self.pe = self.pe.to(data.device)

        return data + self.pe[: data.shape[1]].unsqueeze(0)

    def forward(self, data, output):

        self.pe = self.pe.to(data.device)

        mask = get_mask(data).to(data.device)
        tgt_mask = get_mask(output).to(data.device)

        embedded_data = self._encode_position(self.embed(mask * data))
        embedded_tgt = self._encode_position(self.embed(tgt_mask * output))

        attn_mask = self.transformer.generate_square_subsequent_mask(
            output.shape[1]
        ).to(data.device)

        out = self.transformer(
            embedded_data,
            embedded_tgt,
            src_key_padding_mask=~mask,
            tgt_key_padding_mask=~tgt_mask,
            tgt_mask=attn_mask,
        )
        return self.to_dist(out)

    def inference(self, test_string):
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

        return "".join(tokens[1:-1])
