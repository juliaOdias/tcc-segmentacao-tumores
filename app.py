import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Função para mostrar imagem com máscara sobreposta
def overlay_mask(image, mask, alpha=0.4):
    """Sobrepõe a máscara na imagem em vermelho"""
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2RGB)
    mask_rgb = np.zeros_like(image_rgb)
    mask_rgb[:, :, 0] = mask * 255  # canal vermelho
    
    overlay = cv2.addWeighted(image_rgb, 1.0, mask_rgb, alpha, 0)
    return overlay

st.title("Painel Interativo para Segmentação de Tumores Cerebrais")

uploaded_image = st.file_uploader("Faça upload da imagem MRI (formato PNG ou JPG)", type=['png', 'jpg', 'jpeg'])
uploaded_mask = st.file_uploader("Faça upload da máscara de segmentação (imagem binária PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_image and uploaded_mask:
    image = Image.open(uploaded_image).convert("L")  # abre em escala de cinza
    mask = Image.open(uploaded_mask).convert("L")
    
    # converter máscara para binária (0 ou 1)
    mask_np = np.array(mask)
    mask_bin = (mask_np > 128).astype(np.uint8)
    
    # Mostrar imagem original
    st.subheader("Imagem Original")
    st.image(image, use_column_width=True)
    
    # Mostrar máscara
    st.subheader("Máscara de Segmentação")
    st.image(mask_bin * 255, use_column_width=True)
    
    # Mostrar sobreposição
    st.subheader("Imagem com Máscara Sobreposta")
    overlay_img = overlay_mask(image, mask_bin)
    st.image(overlay_img, use_column_width=True)
    
elif uploaded_image and not uploaded_mask:
    st.warning("Por favor, faça upload também da máscara de segmentação para visualização.")
    
else:
    st.info("Envie as imagens para começar.")
