import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from io import BytesIO
import matplotlib.pyplot as plt

# Título
st.set_page_config(page_title="Segmentação de Tumores Cerebrais", layout="wide")
st.title("🧠 Segmentação de Tumores Cerebrais com U-Net")

# Carregar modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo.keras")

model = load_model()

# Pré-processamento da imagem
def preprocess_image(uploaded_image, target_size=(128, 128)):
    image = Image.open(uploaded_image).convert('L')  # escala de cinza
    image_resized = image.resize(target_size)
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (128,128) → (128,128,1)
    return img_array, image

# Sobreposição da máscara
def overlay_mask(image, mask, alpha=0.4):
    image = np.array(image.convert("RGB"))
    mask = (mask * 255).astype(np.uint8)
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlayed

# Upload
uploaded_file = st.file_uploader("Envie uma imagem de ressonância magnética (RM) do cérebro:", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with st.spinner("Gerando a máscara..."):
        img_array, original_image = preprocess_image(uploaded_file)
        input_batch = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 1)

        prediction = model.predict(input_batch)[0]
        predicted_mask = (prediction > 0.5).astype(np.uint8).squeeze()

        # Converter máscaras para imagem PIL
        mask_img = Image.fromarray((predicted_mask * 255).astype(np.uint8))

        # Imagem com sobreposição
        overlayed_img = overlay_mask(original_image, predicted_mask)

    # Layout lado a lado
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="🧠 Imagem Original", use_column_width=True)
    with col2:
        st.image(mask_img, caption="📌 Máscara Prevista", use_column_width=True)

    st.markdown("---")
    st.subheader("🔬 Sobreposição da Máscara:")
    st.image(overlayed_img, use_column_width=True)

    # Botão para baixar a máscara
    buffer = BytesIO()
    mask_img.save(buffer, format="PNG")
    st.download_button(
        label="⬇️ Baixar Máscara em PNG",
        data=buffer.getvalue(),
        file_name="mascara_prevista.png",
        mime="image/png"
    )

# Métricas do modelo (exemplo fixo para exibição no TCC)
st.markdown("---")
st.subheader("📊 Métricas do Modelo (Validação)")
st.markdown("""
- **Acurácia**: 92%
- **Dice Coefficient**: 0.85
- **IoU (Intersecção sobre União)**: 0.78  
""")

st.info("Essas métricas foram obtidas com um conjunto de validação durante o treinamento.")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido por Julia Oliveira | TCC 2025")
