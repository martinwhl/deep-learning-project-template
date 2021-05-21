import argparse
import torch
import torch.nn as nn


# Adapted from https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py # noqa
class MLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_dim=128, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=str, default=128)
        return parser
