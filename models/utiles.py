import torch
import torch.nn as nn


class FeatureFusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(FeatureFusionMLP, self).__init__()
        self.preprocess_features_weight= nn.Linear(input_dim, input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features, features_weight):

        features_weight = self.preprocess_features_weight(features_weight)
        x = torch.cat([features, features_weight], dim=-1)
        x = self.mlp(x)+features_weight
        return x

