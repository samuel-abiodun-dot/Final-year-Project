# model.py
import torch
import torch.nn as nn
from transformers import RobertaModel
from torchvision import models

class ImageTextMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 768)

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / summed_mask

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pooling(outputs.last_hidden_state, attention_mask)

    def encode_image(self, image_tensor):
        return self.image_encoder(image_tensor)
