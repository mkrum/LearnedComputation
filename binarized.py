import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from rep import (
    BinaryOutputToken,
    BinaryOutputRep,
    MathToken,
    ExpressionRep,
    BinaryVectorRep8bit,
)
from model import BasicModel, VectorInputModel
from dataset import MathDataset, collate_fn
from rich.progress import track
from collections import deque
from main import train

if __name__ == "__main__":
    train(VectorInputModel, BinaryVectorRep8bit, ExpressionRep)
