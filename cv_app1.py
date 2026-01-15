# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd

# -----------------------------
# 1. Page config
# -----------------------------
st.set_page_config(
    page_title="Real-Time Image Classification (ResNet-18)",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Real-Time Image Classification Web App")
st.write("Using **PyTorch ResNet-18 (pretrained on ImageNet)** + Streamlit")

# -----------------------------
# 2. Utility: Load labels
# -----------------------------
@st.cache_data
def load_imagenet_labels():
    """
    Download ImageNet class labels (cached).
    """
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.text.strip().split("\n")

# -----------------------------
# 3. Utility: Load model
# -----------------------------
@st.cache_resource
def load_model():
    """
    Load pretrained ResNet-18 (cached).
    """
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
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 5. Real-time capture (webcam)
# -----------------------------
st.subheader("📸 Capture an Image")
img_file = st.camera_input("Use your camera to take a photo")

# Optional fallback: upload (still acceptable if camera not available)
st.caption("If camera is unavailable, you can upload an image instead.")
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

# Choose whichever input user provides
chosen_file = img_file if img_file is not None else uploaded_file

if chosen_file is not None:
    # Display input image
    image = Image.open(chosen_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    # Preprocess
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # [1, 3, 224, 224]

    # Inference (CPU)
    with torch.no_grad():
        outputs = model(input_batch)
        probs = F.softmax(outputs[0], dim=0)

    # Top-5 predictions
    top5_prob, top5_catid = torch.topk(probs, 5)

    st.subheader("🔍 Top-5 Predictions")
    rows = []
    for i in range(top5_prob.size(0)):
        label = labels[int(top5_catid[i])]
        prob = float(top5_prob[i].item())
        st.write(f"**{label}** — probability: **{prob*100:.2f}%**")
        rows.append({"Label": label, "Probability (%)": round(prob * 100, 2)})

    st.write("### 📊 Predictions Table")
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.caption("Note: These percentages are **model confidence**, not true accuracy (no ground-truth label).")
else:
    st.info("👆 Capture an image (or upload one) to start classification.")
