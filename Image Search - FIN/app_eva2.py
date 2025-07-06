import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from model import ImageTextMatchingModel

device = torch.device("cpu")

# Load model and weights
model = ImageTextMatchingModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Fake evaluation data
def generate_fake_data(batch_size=20):
    images = torch.randn(batch_size, 3, 224, 224)
    ids = torch.randint(0, 1000, (batch_size, 32))
    mask = torch.ones(batch_size, 32)
    return images, ids, mask

images, ids, mask = generate_fake_data()

# Evaluation
with torch.no_grad():
    img_emb = model.encode_image(images.to(device))
    txt_emb = model.encode_text(ids.to(device), mask.to(device))
    similarity = F.cosine_similarity(img_emb, txt_emb)

    predictions = similarity > 0.4
    labels = torch.ones_like(predictions, dtype=torch.bool)

    accuracy = accuracy_score(labels.numpy(), predictions.numpy())
    precision = precision_score(labels.numpy(), predictions.numpy())
    recall = recall_score(labels.numpy(), predictions.numpy())

    # Compute mAP
    similarity_sorted, _ = torch.sort(similarity, descending=True)
    true_positives = torch.ones_like(similarity_sorted)
    precisions = torch.cumsum(true_positives, dim=0) / (torch.arange(1, len(similarity_sorted) + 1))
    mAP = precisions.mean().item()

print(f"Mean Average Precision (mAP): {mAP:.4f}")
