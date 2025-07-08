# 🧠 Segmentação de Tumores Cerebrais com U-Net

Este projeto apresenta uma aplicação interativa baseada em aprendizado profundo para segmentação automática de tumores cerebrais em imagens de ressonância magnética (RM). O modelo foi desenvolvido e treinado como parte do Trabalho de Conclusão de Curso (TCC) em Ciência da Computação.

---

## 🎯 Objetivo

Segmentar regiões de tumores cerebrais a partir de imagens de ressonância magnética utilizando uma rede neural convolucional do tipo **U-Net**, e disponibilizar os resultados de forma visual e acessível por meio de uma aplicação **web interativa com Streamlit**.

---

## 🧪 Tecnologias utilizadas

- [Python](https://www.python.org/) 🐍
- [TensorFlow 2.12.0](https://www.tensorflow.org/)
- [Keras via TensorFlow](https://www.tensorflow.org/guide/keras)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)

---

## 🚀 Como usar a aplicação

### ✔️ Acesse a aplicação online:
[🔗 Clique aqui para acessar no Streamlit Cloud](https://seu-link-aqui.streamlit.app)

### 💻 Para executar localmente:

1. Clone o repositório:

git clone https://github.com/seu-usuario/tumor-segmentation-app.git
cd tumor-segmentation-app

2. Crie e ative um ambiente virtual (opcional):
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3. Instale as dependências:
pip install -r requirements.txt

4. Execute o app:
streamlit run app.py

📸 Funcionalidades da aplicação
📤 Upload de imagem de RM cerebral

🎨 Visualização lado a lado da imagem original e da máscara prevista

🔍 Sobreposição da máscara com transparência sobre a imagem original

⏳ Barra de carregamento durante predição

⬇️ Opção para baixar a máscara prevista em PNG

📊 Exibição de métricas do modelo (acurácia, IoU, Dice)

🌙 Tema escuro com visual profissional e intuitivo

📊 Desempenho do modelo
O modelo foi treinado com imagens segmentadas manualmente e obteve os seguintes resultados em validação:

Acurácia: 92%

Dice Coefficient: 0.85

IoU: 0.78

🧠 Arquitetura utilizada
A rede utilizada foi a U-Net, uma arquitetura amplamente utilizada em segmentação semântica médica. A entrada foi ajustada para (128, 128, 1) com saída binária e função de perda binary_crossentropy.

📁 Estrutura do projeto
tumor-segmentation-app/
├── app.py
├── modelo.keras
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md

👩‍💻 Desenvolvido por
Julia Oliveira
Estudante de Ciência da Computação
TCC 2025 – Segmentação de Imagens Médicas com Deep Learning

📚 Referências
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
https://www.tensorflow.org
https://streamlit.io
https://arxiv.org/abs/1505.04597
