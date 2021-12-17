
import torch

from lc.model import VectorInputModel, BasicModel
from lc.rep import BinaryVectorRep8bit, FloatRep, ExpressionRep

def load_model(model_weights_path, input_rep, output_rep):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = BasicModel
    if input_rep in [FloatRep, BinaryVectorRep8bit]:
        model_type = VectorInputModel

    model = model_type(input_rep, output_rep)

    model.load_state_dict(torch.load(model_weights_path))
    model = model.to(device)
    return model

if __name__ == "__main__":
    model = load_model("models_99.pth", BinaryVectorRep8bit, ExpressionRep)
