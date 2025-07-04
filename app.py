import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Carregar o modelo treinado (compatível com Keras 2.18)
model = load_model("modelo_unet_brain_compat.h5", compile=False)

# Tamanho esperado da imagem
IMG_HEIGHT = 128
IMG_WIDTH = 128

def preprocess_image(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image) / 255.0
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)  # grayscale
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)  # garantir 3 canais
    image = np.expand_dims(image, axis=0)
    return image

def predict_mask(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    mask = prediction[0, :, :, 0]
    return mask

def main():
    st.title("Segmentação de Tumores Cerebrais - U-Net")

    uploaded_file = st.file_uploader("Envie uma imagem de ressonância magnética", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagem Enviada", use_column_width=True)

        if st.button("Segmentar"):
            mask = predict_mask(image)
            st.subheader("Máscara Segmentada")
            st.image(mask, caption="Máscara", use_column_width=True, clamp=True)

if __name__ == "__main__":
    main()
