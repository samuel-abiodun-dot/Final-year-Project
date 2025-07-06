import streamlit as st
import torch
from torchvision import models, transforms
from transformers import RobertaTokenizer, RobertaModel
from PIL import Image
import os
import pickle
import torch.nn as nn

# Model definition
class ImageTextMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 768)

    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output

    def encode_image(self, image_tensor):
        return self.image_encoder(image_tensor)

model = ImageTextMatchingModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load annotation info
with open("annotations.pkl", "rb") as f:
    annotations = pickle.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# UI
st.title("Multimodal Image Search: Text + Image Context")
query = st.text_input("Enter a descriptive text query:")
uploaded_image = st.file_uploader("Upload a query image (optional)", type=["jpg", "jpeg", "png"])

if st.button("Search"):
    if not query and uploaded_image is None:
        st.warning("Please provide at least a text query or an image.")
    else:
        with torch.no_grad():
            # Encode text
            if query:
                text_tokens = tokenizer(query, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
                text_embedding = model.encode_text(text_tokens["input_ids"], text_tokens["attention_mask"])
            else:
                text_embedding = torch.zeros((1, 768))

            # Encode image
            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                image_tensor = transform(image).unsqueeze(0)
                image_embedding = model.encode_image(image_tensor)
            else:
                image_embedding = torch.zeros((1, 768))

            # Combine (average) both embeddings
            joint_query_embedding = (text_embedding + image_embedding) / 2

            # Compare with dataset
            results = []
            for img_name, _ in annotations:
                img_path = os.path.join("images", img_name)
                db_image = Image.open(img_path).convert("RGB")
                db_tensor = transform(db_image).unsqueeze(0)
                db_embedding = model.encode_image(db_tensor)

                similarity = torch.cosine_similarity(joint_query_embedding, db_embedding).item()
                results.append((similarity, img_name))

            results.sort(key=lambda x: x[0], reverse=True)

            # Display results
            st.subheader("Top Matches:")
            for score, name in results[:5]:
                st.image(os.path.join("images", name), caption=f"{name} - Score: {score:.2f}")
