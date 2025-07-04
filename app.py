import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

# Carregar o modelo (.keras)
@st.cache_resource
def load_unet_model():
    return load_model("modelo_unet_brain.keras", compile=False)

model = load_unet_model()

# Configurações da interface
st.set_page_config(page_title="Segmentação de Tumores", layout="centered")
st.title("🧠 Segmentação de Tumores Cerebrais com U-Net")
st.write("Faça upload de uma **imagem de ressonância magnética** (grayscale) para prever a segmentação de tumor.")

# Upload da imagem
uploaded_file = st.file_uploader("📤 Envie a imagem MRI (PNG ou JPG)", type=["png", "jpg", "jpeg"])

# Tamanho esperado pelo modelo
IMG_HEIGHT, IMG_WIDTH = 128, 128

def preprocess_image(image_pil):
    # Converter para escala de cinza
    image = image_pil.convert("L")
    # Redimensionar
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    # Converter para array e normalizar
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=-1)  # (128, 128, 1)
    image = np.expand_dims(image, axis=0)   # (1, 128, 128, 1)
    return image

def overlay_mask(image, mask, alpha=0.5):
    # Converte para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Cria máscara vermelha
    mask_colored = np.zeros_like(image_rgb)
    mask_colored[:, :, 0] = mask  # vermelho no canal R
    # Sobreposição
    blended = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)
    return blended

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Imagem enviada", use_column_width=True)

    # Pré-processamento
    image_input = preprocess_image(image_pil)

    # Predição da máscara
    prediction = model.predict(image_input)[0, :, :, 0]
    mask = (prediction > 0.5).astype(np.uint8) * 255  # binariza e escala para 0-255

    # Sobreposição
    original_resized = np.array(image_pil.convert("L").resize((IMG_WIDTH, IMG_HEIGHT)))
    overlaid = overlay_mask(original_resized, mask)

    # Exibir resultado
    st.subheader("🧪 Resultado da Segmentação")
    st.image(overlaid, caption="Máscara sobreposta à imagem", use_column_width=True)
