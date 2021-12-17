
import argparse
import tqdm
import pickle as pkl

from lc.dataset import FullTestMathDataset
from torch.utils.data import DataLoader
from lc.utils import load_model
from lc.rep import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def get_errors(model, input_rep, output_rep, batch_size=256, num_range=(-128, 127)):
    dataset = FullTestMathDataset(input_rep, output_rep, num_range=(-128, 127))
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=FullTestMathDataset.collate_fn, drop_last=False)
    
    info = {'wrong': [], 'invalid': []}
    total = len(dataset)
    for (x, y, x_raw, y_raw) in tqdm.tqdm(dataloader):
        x = x.cuda()
        output = model.inference(x)

        for i in range(x.shape[0]):
            if output[i] is None:
                info['invalid'].append((x_raw[i], y_raw[i]))

            elif int(y_raw[i]) != int(output[i]):
                info['wrong'].append((x_raw[i], y_raw[i], output[i]))

    return info


def plot_errors(info):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.set_title("Subtraction")
    ax2.set_title("Addition")

    for (x_raw, y_raw, output) in info['wrong']:

        if x_raw[1] == '-':
            ax = ax1
        else:
            ax = ax2

        ax.scatter(int(x_raw[0]), int(x_raw[2]), color='red')

    for (x_raw, y_raw) in info['invalid']:
        if x_raw[1] == '-':
            ax = ax1
        else:
            ax = ax2
        ax.scatter(int(x_raw[0]), int(x_raw[2]), color='black')

    legend_elements = [Line2D([0], [0], marker='o', color='black', label='Invalid'),
                       Line2D([0], [0], marker='o', color='red', label='Wrong')]

    ax2.legend(handles=legend_elements)
  
if __name__ == '__main__':
    model = load_model("float.pth", FloatRep, ExpressionRep)
    out = get_errors(model, FloatRep, ExpressionRep)
    plot_errors(out)
    plt.show()
    pkl.dump(out, open("float_errors.pkl", "wb"))
