import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
WEIGHTS_PATH = "model.weights.h5"
VOCAB_PATH = "vocab.json"
IMG_SIZE = 299
MAX_LEN = 30

st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("🖼️ Image Captioning")

# -------------------------------
# CUSTOM MODEL (SUBCLASSED)
# -------------------------------
class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

# -------------------------------
# BUILD MODEL (MUST MATCH TRAINING)
# -------------------------------
def build_model(vocab_size):
    base_cnn = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet"
    )
    base_cnn.trainable = False

    cnn_model = tf.keras.Sequential([
        base_cnn,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    encoder = tf.keras.layers.Dense(256, activation="relu")

    # IMPORTANT: decoder takes ONLY token ids
    decoder = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 256),
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])

    return ImageCaptioningModel(cnn_model, encoder, decoder)

# -------------------------------
# LOAD MODEL + VOCAB (FORCE BUILD)
# -------------------------------
@st.cache_resource
def load_model_and_vocab():
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    model = build_model(vocab_size)

    # 🔥 FORCE BUILD BEFORE LOADING WEIGHTS
    dummy_image = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    dummy_seq = tf.zeros((1, 1), dtype=tf.int32)

    img_embed = model.cnn_model(dummy_image)
    _ = model.encoder(img_embed)
    _ = model.decoder(dummy_seq)

    model.load_weights(WEIGHTS_PATH)

    return model, word_to_id, id_to_word

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32")
    img = img / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# CAPTION GENERATION (GREEDY)
# -------------------------------
def generate_caption(model, word_to_id, id_to_word, image):
    start_id = word_to_id["<start>"]
    end_id = word_to_id["<end>"]

    image = preprocess_image(image)

    img_embed = model.cnn_model(image)
    _ = model.encoder(img_embed)  # encoder runs, decoder doesn't consume it directly

    caption = [start_id]

    for _ in range(MAX_LEN):
        seq = tf.expand_dims(caption, 0)
        preds = model.decoder(seq)
        next_id = tf.argmax(preds[:, -1, :], axis=-1).numpy()[0]
        caption.append(next_id)

        if next_id == end_id:
            break

    words = [
        id_to_word[i]
        for i in caption
        if i not in [start_id, end_id]
    ]

    return " ".join(words)

# -------------------------------
# UI
# -------------------------------
model, word_to_id, id_to_word = load_model_and_vocab()

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("✨ Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(model, word_to_id, id_to_word, image)
        st.success(caption)
