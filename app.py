import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
@st.cache_resource
def load_unet_model():
    return load_model("modelo_unet_brain.h5", compile=False)

model = load_unet_model()

# Função para sobrepor máscara
def overlay_mask(image, mask, alpha=0.4):
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2RGB)
    mask_rgb = np.zeros_like(image_rgb)
    mask_rgb[:, :, 0] = mask * 255
    return cv2.addWeighted(image_rgb, 1.0, mask_rgb, alpha, 0)

# Layout
st.title("Segmentação de Tumores Cerebrais com U-Net")

uploaded_image = st.file_uploader("Faça upload da imagem MRI (formato PNG ou JPG)", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    image = Image.open(uploaded_image).convert("L")
    st.subheader("Imagem Original")
    st.image(image, use_column_width=True)

    # Pré-processamento da imagem
    img_np = np.array(image.resize((128, 128))) / 255.0
    img_np = np.expand_dims(img_np, axis=(0, -1))  # (1, 128, 128, 1)

    # Prever a máscara com o modelo
    prediction = model.predict(img_np)[0]
    mask_pred = (prediction > 0.5).astype(np.uint8).squeeze()

    # Exibir máscara prevista
    st.subheader("Máscara Prevista")
    st.image(mask_pred * 255, use_column_width=True)

    # Sobreposição
    st.subheader("Imagem com Máscara Sobreposta")
    overlay = overlay_mask(image.resize((128, 128)), mask_pred)
    st.image(overlay, use_column_width=True)
else:
    st.info("Envie uma imagem de ressonância magnética para segmentar.")
