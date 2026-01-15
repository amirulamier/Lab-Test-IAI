import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import json
from io import BytesIO

# -----------------------------
# 1. Page config
# -----------------------------
st.set_page_config(
    page_title="Image Classification with PyTorch & Streamlit",
    page_icon="",
    layout="centered"
)

st.title("Simple Image Classification Web App")
st.write("Using **PyTorch ResNet-18 (pretrained on ImageNet)** + Streamlit")

# -----------------------------
# 2. Utility: Load labels
# -----------------------------
@st.cache_data
def load_imagenet_labels():
    """
    Download ImageNet class index (only once, then cached).
    """
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.strip().split("\n")
    return labels

# -----------------------------
# 3. Utility: Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

labels = load_imagenet_labels()
model = load_model()

# -----------------------------
# 4. Define transforms
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # Normalization for ImageNet pretrained models
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 5. File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (jpg/png)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]

    # Move to device (CPU only for simplicity)
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs[0], dim=0)

    # Get top-5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("🔍 Top-5 Predictions")
    for i in range(top5_prob.size(0)):
        st.write(
            f"**{labels[top5_catid[i]]}** — "
            f"probability: {top5_prob[i].item():.4f}"
        )

    # Show as table
    st.write("### 📊 Predictions Table")
    import pandas as pd

    df = pd.DataFrame({
        "Label": [labels[idx] for idx in top5_catid],
        "Probability": [float(p) for p in top5_prob]
    })
    st.dataframe(df, use_container_width=True)

else:
    st.info("👆 Please upload an image to start.")
