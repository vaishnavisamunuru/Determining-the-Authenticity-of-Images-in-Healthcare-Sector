# app.py
import streamlit as st
import torch
import json
from torchvision import transforms
from PIL import Image
import io

from models import cnn_model,resnet_model,densenet_model

# 1) Page setup
st.set_page_config(page_title="Medical Image Tamper Detector", layout="centered")
st.title("🩺 Medical Image Tamper Detector")

# 2) Load metadata
@st.cache_resource
def load_metadata():
    with open("models/metadata.json","r") as f:
        return json.load(f)
meta = load_metadata()


# 4) Load models into a dict
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    # CNN
    cnn_model.load_state_dict(torch.load(meta["cnn_path"], map_location=device))
    models["SimpleCNN"] = cnn_model.to(device).eval()
    # ResNet
    rnet = resnet_model
    rnet.load_state_dict(torch.load(meta["resnet_path"], map_location=device))
    models["ResNet-50"] = rnet.to(device).eval()
    # DenseNet
    dnet = densenet_model
    dnet.load_state_dict(torch.load(meta["densenet_path"], map_location=device))
    models["DenseNet-121"] = dnet.to(device).eval()
    return models, device

models, device = load_models()

# 5) Sidebar: choose model
model_name = st.sidebar.selectbox("Choose model", list(models.keys()))
model = models[model_name]

# 6) Upload & display image
st.header("Upload a CT Scan/ X-ray / MRI")
uploaded = st.file_uploader("", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(img, caption="Input Image", width=200)

    # 7) Preprocess (match your training)
    preprocess = transforms.Compose([
        transforms.Resize((28,28)),  # same as training
        transforms.ToTensor(),         # scales to [0,1]
    ])
    inp = preprocess(img).unsqueeze(0).to(device)

    # 8) Predict
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = int(probs.argmax())

    labels = meta["classes"]
    st.markdown(f"**Model**: {model_name}")
    st.markdown(f"**Prediction:** {labels[pred]}")
    st.markdown(f"**P(tampered):** {probs[1]:.4f}")
