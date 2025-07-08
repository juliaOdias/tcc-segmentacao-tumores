import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

# === Carregar o modelo ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("modelo.keras")
    return model

model = load_model()

# === Função para pré-processar imagem ===
def preprocess_image(uploaded_image, target_size=(128, 128)):
    image = Image.open(uploaded_image).convert('L')  # <- 'L' força escala de cinza
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0  # normalização
    img_array = np.expand_dims(img_array, axis=-1)  # (128,128) -> (128,128,1)
    return img_array, image


# === Função para prever a máscara ===
def predict_mask(model, img_array):
    input_image = np.expand_dims(img_array, axis=0)  # Adiciona batch dimension
    pred_mask = model.predict(input_image)[0]  # Remove batch
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Binariza
    return pred_mask

# === Interface do usuário ===
st.title("Segmentação de Tumores Cerebrais com CNN")
st.write("Faça o upload de uma imagem de ressonância para segmentação do tumor.")

uploaded_file = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_array, original_image = preprocess_image(uploaded_file)
    st.image(original_image, caption="Imagem Original", use_column_width=True)

    st.write("Gerando máscara...")
    pred_mask = predict_mask(model, img_array)

    # Mostrar máscara prevista
    st.image(pred_mask, caption="Máscara Prevista", use_column_width=True, clamp=True)
