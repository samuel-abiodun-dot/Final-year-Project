import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
#pip install torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pickle

# Dataset
class ImageTextDataset(Dataset):
    def __init__(self, image_dir, annotations, tokenizer, transform):
        self.image_dir = image_dir
        self.annotations = annotations  # list of (filename, caption)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, caption = self.annotations[idx]
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        image = self.transform(image)
        encoding = self.tokenizer(caption, return_tensors='pt', padding='max_length',
                                  truncation=True, max_length=32)
        return image, encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

# Model
class ImageTextMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 768)
        self.fc = nn.CosineSimilarity(dim=1)

    def forward(self, image, input_ids, attention_mask):
        text_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_embeds = self.image_encoder(image)
        similarity = self.fc(image_embeds, text_embeds)
        return similarity, image_embeds, text_embeds

# Training loop (simple pairwise match)
def train():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Custom dataset for illustration
    annotations = [
                   ('cat.jpg', 'A cat sitting on a sofa'),
                   ('cow.jpg', 'A black and white cow grazing in a field')
                   ]
    dataset = ImageTextDataset('./images', annotations, tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=2)

    model = ImageTextMatchingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(2):
        for img, ids, mask in dataloader:
            optimizer.zero_grad()
            similarity, img_emb, txt_emb = model(img, ids, mask)
            target = torch.ones(similarity.size())
            loss = criterion(similarity, target)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    with open('annotations.pkl', 'wb') as f:
        pickle.dump(annotations, f)

if __name__ == "__main__":
    train()
