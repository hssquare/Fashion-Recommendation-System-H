import os
import pickle

import numpy as np
import streamlit as st
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import load_img, img_to_array


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("Fashion Recommendation System")


# ---------------------------------------------------------
# Load saved feature database
# ---------------------------------------------------------

with open("image_features_embedding.pkl", "rb") as f:
    features_list = pickle.load(f)

with open("img_files.pkl", "rb") as f:
    img_files_list = pickle.load(f)


# ---------------------------------------------------------
# Load and cache ResNet50 model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    return Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])


model = load_model()


# ---------------------------------------------------------
# Number of recommendations
# ---------------------------------------------------------

top_k = st.selectbox(
    "Number of recommendations",
    options=[5, 10, 15],
    index=0
)


# ---------------------------------------------------------
# Build nearest-neighbor index
# ---------------------------------------------------------

neighbors = NearestNeighbors(
    n_neighbors=top_k + 1,
    algorithm="brute",
    metric="euclidean"
)

neighbors.fit(features_list)


# ---------------------------------------------------------
# Save uploaded image
# ---------------------------------------------------------

def save_file(uploaded_file):
    try:
        os.makedirs("uploader", exist_ok=True)

        file_path = os.path.join(
            "uploader",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    except Exception as exc:
        st.error(
            f"Could not save uploaded image: {exc}"
        )
        return None


# ---------------------------------------------------------
# Extract features from query image
# ---------------------------------------------------------

def extract_img_features(img_path, model):
    img = load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = img_to_array(img)

    expand_img = np.expand_dims(
        img_array,
        axis=0
    )

    preprocessed_img = preprocess_input(
        expand_img
    )

    result_to_resnet = model.predict(
        preprocessed_img,
        verbose=0
    )

    flatten_result = result_to_resnet.flatten()

    feature_norm = norm(flatten_result)

    if feature_norm == 0:
        return flatten_result

    normalized_result = (
        flatten_result / feature_norm
    )

    return normalized_result


# ---------------------------------------------------------
# Find similar images
# ---------------------------------------------------------

def recommend(features):
    distances, indices = neighbors.kneighbors(
        [features]
    )

    return distances[0], indices[0]


# ---------------------------------------------------------
# Upload image
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose your image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    file_path = save_file(uploaded_file)

    if file_path:

        # ---------------------------------------------
        # Display uploaded image
        # ---------------------------------------------

        show_image = Image.open(uploaded_file)

        resized_image = show_image.resize(
            (400, 400)
        )

        st.image(
            resized_image,
            caption="Uploaded Image"
        )

        # ---------------------------------------------
        # Extract query features
        # ---------------------------------------------

        features = extract_img_features(
            file_path,
            model
        )

        # ---------------------------------------------
        # Get recommendations
        # ---------------------------------------------

        distances, indices = recommend(
            features
        )

        # ---------------------------------------------
        # Display recommendations
        # ---------------------------------------------

        st.subheader(
            f"Top {top_k} Recommended Products"
        )

        # Display results in rows of 5
        for start in range(0, top_k, 5):

            row_positions = list(
                range(
                    start,
                    min(start + 5, top_k)
                )
            )

            columns = st.columns(
                len(row_positions)
            )

            for col, position in zip(
                columns,
                row_positions
            ):
                with col:

                    st.image(
                        img_files_list[
                            indices[position]
                        ]
                    )

                    st.caption(
                        f"Distance: "
                        f"{distances[position]:.4f}"
                    )

    else:
        st.error(
            "Could not save the uploaded image."
        )