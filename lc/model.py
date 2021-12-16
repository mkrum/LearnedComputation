import math
import torch
import torch.nn as nn
from lc.rep import MathToken, BinaryOutputToken, BinaryOutputRep
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
    def __init__(self, input_expression, output_expression):
        super().__init__()

        self.input_expression = input_expression
        self.output_token = output_expression.token_type
        self.output_expression = output_expression

        self.embed = nn.Embedding(self.output_token.size(), 128)
        self.transformer = nn.Transformer(
            num_encoder_layers=6, num_decoder_layers=6, batch_first=True, d_model=128
        )
        self.pe = positionalencoding1d(128, 1000)
        self.to_dist = nn.Sequential(nn.Linear(128, self.output_token.size()))

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

    def inference(self, x, max_len=5):
        act = nn.LogSoftmax(dim=-1)

        tokens = [["<start>"] for _ in range(x.shape[0])]

        for _ in range(max_len):

            out = torch.tensor(
                [
                    [
                        self.output_token.from_str(tokens[j][i]).to_int()
                        for i in range(len(tokens[j]))
                    ]
                    for j in range(x.shape[0])
                ]
            )
            out = out.cuda()

            with torch.no_grad():
                logits = self.forward(x, out)[:, -1, :]

            preds = torch.argmax(logits, -1)
            next_tokens = [
                self.output_token.from_int(preds[i].item()).to_str()
                for i in range(x.shape[0])
            ]
            for (i, t) in enumerate(next_tokens):
                tokens[i].append(t)

        out = []
        for t in tokens:
            out.append(self.output_expression.parse(t))

        return out


class VectorInputModel(BasicModel):
    def __init__(self, input_expression, output_expression):

        super().__init__(input_expression, output_expression)
        self.embed_fc = nn.Sequential(
            nn.Linear(input_expression.size(), 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 128),
        )

    def forward(self, data, output):

        self.pe = self.pe.to(data.device)

        mask = get_mask(data).to(data.device)
        tgt_mask = get_mask(output).to(data.device)

        embedded_data = self._encode_position(self.embed_fc(data))
        embedded_tgt = self._encode_position(self.embed(tgt_mask * output))

        attn_mask = self.transformer.generate_square_subsequent_mask(
            output.shape[1]
        ).to(data.device)
        out = self.transformer(
            embedded_data,
            embedded_tgt,
            src_key_padding_mask=~mask[:, :, 0].view(mask.shape[0], mask.shape[1]),
            tgt_key_padding_mask=~tgt_mask,
            tgt_mask=attn_mask,
        )
        return self.to_dist(out)
