import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Carregar modelo (certifique-se que o arquivo .keras está no mesmo diretório)
@st.cache_resource(show_spinner=True)
def load_unet_model():
    model = load_model("modelo_unet_brain_compat.keras", compile=False)
    return model

model = load_unet_model()

st.title("Segmentação com modelo U-Net")

uploaded_file = st.file_uploader("Envie uma imagem 128x128 em grayscale", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Abrir imagem e converter para grayscale e tamanho correto
    image = Image.open(uploaded_file).convert("L").resize((128, 128))
    st.image(image, caption="Imagem enviada", use_column_width=True)

    # Preparar array para predição
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape((1, 128, 128, 1))  # batch_size, height, width, channels

    if st.button("Predizer segmentação"):
        preds = model.predict(image_array)
        # Supondo que saída seja máscara com shape (1, 128, 128, 1)
        mask = preds[0, :, :, 0]
        # Normalizar para exibir melhor (0-255)
        mask_img = (mask * 255).astype(np.uint8)
        st.image(mask_img, caption="Máscara predita", use_column_width=True)
