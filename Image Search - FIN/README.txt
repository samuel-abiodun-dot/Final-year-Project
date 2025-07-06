
# 📷 Enhancing Image Search Results Using Pre-trained NLP Models

## 🧠 Overview

This project showcases a **multimodal image search system** that integrates **Transformer-based NLP models** (like RoBERTa) with **deep learning-based CNN models** (like ResNet50) to improve the accuracy and relevance of image search results.

The system uses a **Streamlit-based web interface**, enabling users to search for images using either **text descriptions** or **image uploads**. Behind the scenes, a shared embedding space maps both modalities for semantic alignment and retrieval.

---

## 🧰 Tools and Technologies

- **Language Model**: RoBERTa (via Hugging Face Transformers)
- **Image Model**: ResNet50 (via TorchVision)
- **Interface**: Streamlit
- **Programming Language**: Python 3.8+
- **IDE**: PyCharm or VS Code
- **Core Libraries**: PyTorch, Transformers, torchvision, PIL, scikit-learn

---

## ⚙️ System Requirements

- Minimum: 8GB RAM, 500GB HDD, Core i3 CPU  
- Recommended: GPU (8GB) for model fine-tuning and faster inference

---

## 📂 Dataset

- A **custom dataset** is used, consisting of images and their corresponding natural language captions.
- Place all images in a folder named `images/`, and pair them with their captions in a Python list or JSON file (`annotations.pkl`).

---

## 🧪 How to Run the Project


### 1. 🧱 Set Up the Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

 ### 3. 🔗 Download Required File

This project requires a file that is too large to be stored in the repository. You can download it from the following Google Drive link:

**[Download File from Google Drive](https://drive.google.com/file/d/1mZCdnyDLL8c-toWvzJ6pudh4Bh0Nah-k/view?usp=drive_link)**

### 📂 Where to Place the File

After downloading, place the file in the following directory relative to the project root:Image Search - FIN



### 4. 🖼️ Prepare the Dataset

- Create a folder `images/` and place your image files (e.g., `cat.jpg`, `dog.jpg`) inside.
- Create an `annotations.pkl` file containing a list of tuples like:

```python
[("cat.jpg", "A cat sitting on a sofa"), ("cow.jpg", "A black and white cow in the field")]
```

Use the included training script to generate this:

```bash
python train_model.py
```

This script:
- Encodes the image-text pairs
- Trains a simple embedding model
- Saves the model as `model.pth`

### 5. 🚀 Run the Web App

```bash
streamlit run app.py
```

- You’ll see the app running in your browser at `http://localhost:8501/`
- Enter a **text query** or **upload an image**, and the app will return the top matching results with similarity scores.

---

## 📊 Evaluation Metrics

| Input Type       | Cosine Similarity | Mean Average Precision (mAP) |
|------------------|-------------------|------------------------------|
| Text Query       | 0.49              | 1.00                         |
| Image Upload     | 1.00              | 1.00                         |

---

## 🔮 Future Enhancements

- Incorporate larger public datasets (e.g., MS-COCO)
- Enable GPU acceleration
- Add feedback learning for personalized retrieval
- Fine-tune on domain-specific datasets

---

## 💼 Applications

- 🛒 E-commerce product search  
- 🏥 Medical image retrieval  
- 🧑‍🏫 Education and digital learning  
- 📷 Creative asset management  
- 🔍 Social media tagging & moderation  

---

## 👨‍💻 Author

**Abiodun Samuel Mofiyinfoluwa**  
Department of Computer Science  
Caleb University, Imota, Lagos State, Nigeria  
Final Year Project (2024)

---

---
