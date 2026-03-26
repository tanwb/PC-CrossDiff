import torch
import torch.nn as nn
import re
from .attribute_object_extract import *

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


def contains_letter(s):
    # Search for any alphabetic character (case-insensitive); return True if found
    return bool(re.search('[a-zA-Z]', s))


def is_empty(element):
    if element is None:
        return True
    if isinstance(element, str) and element == "":
        return True
    if isinstance(element, list) and not element:
        return True
    if isinstance(element, dict) and not element:
        return True
    if not contains_letter(element):
        return True
    return False

def pre_processed(input_text, input_attribute):
    processed_attribute_object = []
    for i in range(len(input_attribute)):
        if is_empty(input_attribute[i]):
            replacement_value = input_text[i]
            print(f"Replacing empty attribute at index {i} with text value: {replacement_value}")  # Debug print
            processed_attribute_object.append(replacement_value)
        else:
            processed_attribute_object.append(attribute_object(input_attribute[i]))
    return processed_attribute_object

