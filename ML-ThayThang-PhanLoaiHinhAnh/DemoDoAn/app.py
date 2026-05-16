# =====================================================
# 🧠 IMAGE CLASSIFICATION STREAMLIT APP
# =====================================================
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import vit_b_16
from PIL import Image
import time
import os
import pandas as pd
import plotly.express as px

# =====================================================
# 1️⃣ CẤU HÌNH BAN ĐẦU
# =====================================================
st.set_page_config(page_title="Image Classification Demo", layout="wide")
st.title("🧠 Image Classification - Compare 5 Models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
IMG_SIZE = 224
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# =====================================================
# 2️⃣ CHUẨN BỊ TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# 3️⃣ HÀM XÂY DỰNG MÔ HÌNH
# =====================================================
def build_model(model_name, num_classes=6):
    name = model_name.lower()
    if name == "vgg16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    elif name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    elif name == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    elif name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    elif name == "vit":
        model = vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)

# =====================================================
# 4️⃣ LOAD TOÀN BỘ MÔ HÌNH
# =====================================================
@st.cache_resource
def load_all_models():
    models_dict = {}
    model_files = {
        "VGG16": "vgg16_best.pt",
        "ResNet50": "resnet50_best.pt",
        "MobileNet": "mobilenet_best.pt",
        "EfficientNet": "efficientnet_best.pt",
        "ViT": "vit_best.pt"
    }

    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            st.error(f"⚠️ Model file not found: {path}")
            continue

        model = build_model(name, num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models_dict[name] = model
        st.sidebar.success(f"✅ Loaded {name}")

    return models_dict

models_dict = load_all_models()

# =====================================================
# 5️⃣ HÀM DỰ ĐOÁN ẢNH
# =====================================================
def predict_image(model, image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        start_time = time.time()
        outputs = model(img_tensor)
        end_time = time.time()
        probs = torch.softmax(outputs, dim=1)[0]
        _, preds = torch.max(probs, 0)
        label = CLASS_NAMES[preds.item()]
        confidence = probs[preds].item()
        return label, confidence, (end_time - start_time)

# =====================================================
# 6️⃣ UPLOAD ẢNH & HIỂN THỊ KẾT QUẢ
# =====================================================
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])  # 2 cột: ảnh & kết quả

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("🔍 Predictions from Models")
        results = []

        with st.spinner("⏳ Predicting across all models..."):
            for name, model in models_dict.items():
                label, conf, t_inf = predict_image(model, image)
                results.append({
                    "Model": name,
                    "Predicted Class": label,
                    "Confidence (%)": conf * 100,
                    "Inference Time (s)": t_inf
                })

        df = pd.DataFrame(results)

        # =====================================================
        # 📊 BẢNG KẾT QUẢ
        # =====================================================
        st.markdown("### 📊 Inference Results")
        st.dataframe(df.style.format({
            "Confidence (%)": "{:.2f}",
            "Inference Time (s)": "{:.3f}"
        }), use_container_width=True)

        # =====================================================
        # 📈 BIỂU ĐỒ 1: ĐỘ TIN CẬY
        # =====================================================
        st.markdown("### 📈 Confidence Comparison Among Models")
        fig_conf = px.bar(
            df,
            x="Model",
            y="Confidence (%)",
            color="Model",
            text=df["Confidence (%)"].apply(lambda x: f"{x:.2f}%"),
            title="Model Confidence Comparison",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_conf.update_traces(textposition='outside')
        fig_conf.update_layout(yaxis_range=[0, 110])
        st.plotly_chart(fig_conf, use_container_width=True)

        # =====================================================
        # ⚡ BIỂU ĐỒ 2: THỜI GIAN SUY LUẬN
        # =====================================================
        st.markdown("### ⚡ Inference Speed Comparison (seconds)")
        fig_time = px.bar(
            df,
            x="Model",
            y="Inference Time (s)",
            color="Model",
            text=df["Inference Time (s)"].apply(lambda x: f"{x:.3f}s"),
            title="Model Inference Time Comparison",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_time.update_traces(textposition='outside')
        fig_time.update_layout(yaxis_range=[0, df["Inference Time (s)"].max() * 1.2])
        st.plotly_chart(fig_time, use_container_width=True)

        # =====================================================
        # 🏆 GỢI Ý MÔ HÌNH TỐT NHẤT
        # =====================================================
        best_row = df.loc[df["Confidence (%)"].idxmax()]
        st.success(
            f"🏆 **Best Prediction:** {best_row['Model']} → "
            f"{best_row['Predicted Class']} ({best_row['Confidence (%)']:.2f}%)"
        )

else:
    st.info("⬆️ Please upload an image file to begin classification.")
