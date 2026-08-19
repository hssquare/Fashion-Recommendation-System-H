import pickle

import numpy as np
import streamlit as st
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👕",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("Fashion Recommendation System")

st.write(
    "Upload a fashion/product image to find visually similar products."
)


# =========================================================
# LOAD FEATURE DATABASE
# =========================================================

@st.cache_data
def load_feature_database():
    with open("image_features_embedding.pkl", "rb") as f:
        features = pickle.load(f)

    with open("img_files.pkl", "rb") as f:
        image_files = pickle.load(f)

    return np.asarray(features), image_files


features_list, img_files_list = load_feature_database()


# =========================================================
# LOAD AND CACHE RESNET50
# =========================================================

@st.cache_resource
def load_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Use ResNet50 only as a frozen feature extractor
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    return model


model = load_model()


# =========================================================
# NUMBER OF RECOMMENDATIONS
# =========================================================

top_k = st.selectbox(
    "Number of recommendations",
    options=[5, 10, 15],
    index=0
)


# =========================================================
# BUILD NEAREST NEIGHBOR INDEX
# =========================================================

neighbors = NearestNeighbors(
    n_neighbors=top_k + 1,
    algorithm="brute",
    metric="euclidean"
)

neighbors.fit(features_list)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_img_features(img, model):
    """
    Extract a normalized ResNet50 feature vector
    from a PIL image.
    """

    # Ensure RGB
    img = img.convert("RGB")

    # ResNet50 input size
    img = img.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    # Convert image to NumPy array
    img_array = np.asarray(
        img,
        dtype=np.float32
    )

    # Add batch dimension
    expanded_img = np.expand_dims(
        img_array,
        axis=0
    )

    # ResNet50 preprocessing
    preprocessed_img = preprocess_input(
        expanded_img
    )

    # Feature extraction
    result = model.predict(
        preprocessed_img,
        verbose=0
    )

    # Flatten embedding
    feature_vector = result.flatten()

    # L2 normalization
    feature_norm = norm(feature_vector)

    if feature_norm == 0:
        return feature_vector

    normalized_vector = (
        feature_vector / feature_norm
    )

    return normalized_vector


# =========================================================
# RECOMMENDATION
# =========================================================

def recommend(features):
    """
    Find nearest images using Euclidean distance.
    """

    distances, indices = neighbors.kneighbors(
        [features]
    )

    return distances[0], indices[0]


# =========================================================
# UPLOAD IMAGE
# =========================================================

uploaded_file = st.file_uploader(
    "Choose your image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# =========================================================
# PROCESS UPLOADED IMAGE
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # Read image directly into memory
        # -------------------------------------------------

        query_image = Image.open(
            uploaded_file
        ).convert("RGB")

        # -------------------------------------------------
        # Display uploaded image
        # -------------------------------------------------

        st.subheader("Uploaded Image")

        st.image(
            query_image,
            width=400
        )

        # -------------------------------------------------
        # Extract query features
        # -------------------------------------------------

        features = extract_img_features(
            query_image,
            model
        )

        # -------------------------------------------------
        # Find nearest images
        # -------------------------------------------------

        distances, indices = recommend(
            features
        )

        # -------------------------------------------------
        # Remove exact filename match if possible
        # -------------------------------------------------

        uploaded_name = (
            uploaded_file.name
            .lower()
        )

        valid_results = []

        for distance, index in zip(
            distances,
            indices
        ):

            image_path = str(
                img_files_list[index]
            )

            # Windows/Linux path compatibility
            image_name = image_path.replace(
                "\\",
                "/"
            ).split("/")[-1].lower()

            if image_name == uploaded_name:
                continue

            valid_results.append(
                (distance, index)
            )

            if len(valid_results) == top_k:
                break

        # -------------------------------------------------
        # Results title
        # -------------------------------------------------

        st.subheader(
            f"Top {len(valid_results)} Recommended Products"
        )

        # -------------------------------------------------
        # Display results in rows of 5
        # -------------------------------------------------

        for start in range(
            0,
            len(valid_results),
            5
        ):

            row = valid_results[
                start:start + 5
            ]

            columns = st.columns(
                len(row)
            )

            for col, (
                distance,
                index
            ) in zip(
                columns,
                row
            ):

                with col:

                    image_path = (
                        img_files_list[index]
                    )

                    # -----------------------------
                    # Open original image
                    # -----------------------------

                    result_image = Image.open(
                        image_path
                    ).convert("RGB")

                    # -----------------------------
                    # Preserve aspect ratio
                    # -----------------------------

                    result_image.thumbnail(
                        (180, 220),
                        Image.Resampling.LANCZOS
                    )

                    # -----------------------------
                    # Display image without
                    # stretching
                    # -----------------------------

                    st.image(
                        result_image,
                        width=180
                    )

                    # -----------------------------
                    # Distance
                    # -----------------------------

                    st.caption(
                        f"Distance: {distance:.4f}"
                    )

    except Exception as exc:

        st.error(
            f"Could not process the uploaded image: {exc}"
        )