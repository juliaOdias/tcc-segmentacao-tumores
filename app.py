import streamlit as st
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Segmentação de Tumores Cerebrais", layout="wide")

# Sidebar com explicação
with st.sidebar:
    st.title("Sobre este App")
    st.write("""
        Este app usa um modelo U-Net treinado para segmentação de tumores cerebrais em imagens médicas.
        Você pode enviar uma ou várias imagens, e verá a segmentação ao lado da imagem original,
        além de poder baixar a máscara resultante.
    """)
    st.write("---")

@st.cache_resource
def carregar_modelo(caminho_modelo):
    model = tf.keras.models.load_model(caminho_modelo)
    return model

model = carregar_modelo("modelo_unet_brain_compat.h5")

def preprocessar(imagem_pil):
    img = imagem_pil.convert('L').resize((128,128))
    img_arr = np.array(img) / 255.0
    img_arr = img_arr[np.newaxis, ..., np.newaxis]  # Formato batch, altura, largura, canal
    return img_arr.astype(np.float32)

def predizer_mascara(model, img_prep):
    pred = model.predict(img_prep)[0, :, :, 0]
    pred_bin = (pred > 0.5).astype(np.uint8)
    return pred_bin

def sobrepor_mascara(imagem, mascara_bin):
    img_rgb = imagem.convert("RGB").resize((128,128))
    mask_color = np.zeros((128,128,3), dtype=np.uint8)
    mask_color[..., 0] = 255 * mascara_bin  # Vermelho para máscara
    img_np = np.array(img_rgb)
    sobreposicao = img_np.copy()
    alpha = 0.4
    mask_on = mask_color > 0
    sobreposicao[mask_on] = (1 - alpha) * img_np[mask_on] + alpha * mask_color[mask_on]
    sobreposicao = sobreposicao.astype(np.uint8)
    return Image.fromarray(sobreposicao)

def main():
    st.title("Segmentação de Tumores Cerebrais")
    st.write("Envie imagens para segmentar os tumores. Você pode enviar múltiplas imagens.")

    uploaded_files = st.file_uploader("Escolha imagens (PNG, JPG)", type=['png','jpg','jpeg'], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Por favor, envie pelo menos uma imagem.")
        return

    dice_exemplo = 0.87  # Exemplo estático, substitua por métrica real se disponível

    for idx, file in enumerate(uploaded_files):
        st.markdown(f"### Imagem {idx+1}: {file.name}")
        imagem_pil = Image.open(file)

        with st.spinner("Processando imagem..."):
            img_prep = preprocessar(imagem_pil)
            mascara_bin = predizer_mascara(model, img_prep)
            mascara_img = Image.fromarray(mascara_bin * 255).convert("L")
            img_sobreposta = sobrepor_mascara(imagem_pil, mascara_bin)

        # Layout em colunas
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.image(imagem_pil, caption="Imagem Original", use_column_width=True)
        with col2:
            st.image(mascara_img, caption="Máscara Segmentada (Binária)", use_column_width=True)
        with col3:
            st.image(img_sobreposta, caption="Imagem com Máscara Sobreposta", use_column_width=True)

        # Mostrar métrica (exemplo)
        st.metric(label="Dice Coefficient (exemplo)", value=f"{dice_exemplo:.2f}")

        # Botão para download da máscara
        buf = io.BytesIO()
        mascara_img.save(buf, format="PNG")
        st.download_button(
            label="Baixar Máscara (PNG)",
            data=buf.getvalue(),
            file_name=f"mascara_{file.name}.png",
            mime="image/png"
        )
        st.write("---")

if __name__ == "__main__":
    main()
