import json
import pickle
import time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
)
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fashion Vision",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

FEATURES_FILE = "image_features_embedding.pkl"
IMAGE_FILES_FILE = "img_files.pkl"
HF_MAPPING_FILE = "huggingface_product_mapping.json"

# Your own Hugging Face dataset
HF_DATASET = "GangHitman/fashion-recommendation-images"
HF_CONFIG = "default"
HF_SPLIT = "train"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"

APP_MODE_LABEL = "FULL"
APP_MODE_DESCRIPTION = "44,441-image retrieval catalog"


# =========================================================
# HUGGING FACE AUTHENTICATION
# =========================================================

try:
    HF_TOKEN = st.secrets.get(
        "HF_TOKEN",
        "",
    ).strip()
except Exception:
    HF_TOKEN = ""

HF_HEADERS = {
    "User-Agent": "FashionVision/1.0"
}

if HF_TOKEN:
    HF_HEADERS["Authorization"] = (
        f"Bearer {HF_TOKEN}"
    )


# =========================================================
# HTML HELPER
# =========================================================

def render_html(html_content):
    st.html(html_content)


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(105, 62, 255, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(0, 210, 255, 0.11),
            transparent 27%
        ),
        radial-gradient(
            circle at 70% 90%,
            rgba(255, 0, 153, 0.09),
            transparent 28%
        ),
        #06070b;
    color: #f7f7fb;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}


/* =========================================================
   NAVIGATION
   ========================================================= */

.fv-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 22px;
    margin-bottom: 75px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.015)
        );
    backdrop-filter: blur(24px);
    box-shadow:
        0 20px 80px rgba(0,0,0,0.36);
}

.fv-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.fv-logo {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background:
        linear-gradient(
            135deg,
            #7d3cff,
            #00d5ff
        );
    font-size: 19px;
    font-weight: 900;
    box-shadow:
        0 0 45px rgba(100,60,255,0.38);
}

.fv-brand-text {
    color: white;
    font-size: 15px;
    font-weight: 900;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.fv-online {
    display: inline-flex;
    align-items: center;
    padding: 8px 14px;
    border-radius: 999px;
    background:
        rgba(0,255,180,0.055);
    border:
        1px solid rgba(0,255,180,0.18);
    color: #6fffd0;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}

.fv-online-dot {
    width: 7px;
    height: 7px;
    margin-right: 7px;
    border-radius: 50%;
    background: #43ff9d;
    box-shadow:
        0 0 14px #43ff9d;
}


/* =========================================================
   HERO
   ========================================================= */

.fv-hero {
    padding: 10px 0 90px;
}

.fv-eyebrow {
    display: inline-flex;
    padding: 8px 13px;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 999px;
    background:
        rgba(255,255,255,0.035);
    color: #9699aa;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.17em;
    text-transform: uppercase;
}

.fv-hero-title {
    margin-top: 25px;
    max-width: 1000px;
    font-size: clamp(62px, 8vw, 126px);
    line-height: 0.86;
    font-weight: 950;
    letter-spacing: -0.075em;
    background:
        linear-gradient(
            100deg,
            #ffffff 0%,
            #ffffff 28%,
            #c9b9ff 50%,
            #62e8ff 74%,
            #ff65c7 100%
        );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.fv-hero-description {
    max-width: 690px;
    margin-top: 30px;
    color: #9fa2b1;
    font-size: 17px;
    line-height: 1.75;
}


/* =========================================================
   SECTION HEADINGS
   ========================================================= */

.fv-section-eyebrow {
    color: #888b9c;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.fv-section-title {
    margin-top: 9px;
    color: #f8f8fd;
    font-size: clamp(34px, 4vw, 55px);
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: -0.055em;
}

.fv-section-description {
    margin-top: 13px;
    color: #858897;
    font-size: 13px;
    line-height: 1.6;
}


/* =========================================================
   MODE CARD
   ========================================================= */

.fv-mode-card {
    margin-top: 30px;
    padding: 15px 18px;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    background:
        rgba(255,255,255,0.022);
    color: #888b9c;
    font-size: 11px;
    line-height: 1.6;
}

.fv-mode-value {
    color: #6fffd0;
    font-weight: 900;
}


/* =========================================================
   UPLOAD
   ========================================================= */

.fv-upload {
    margin-top: 32px;
    padding: 65px 30px;
    text-align: center;
    border:
        1px dashed rgba(126,88,255,0.62);
    border-radius: 30px;
    background:
        radial-gradient(
            circle at center,
            rgba(111,65,255,0.11),
            transparent 60%
        ),
        rgba(255,255,255,0.015);
    box-shadow:
        inset 0 0 90px rgba(108,65,255,0.025);
}

.fv-upload-icon {
    width: 74px;
    height: 74px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 21px;
    background:
        linear-gradient(
            135deg,
            rgba(124,60,255,0.27),
            rgba(0,212,255,0.15)
        );
    border:
        1px solid rgba(255,255,255,0.09);
    color: white;
    font-size: 28px;
    box-shadow:
        0 0 55px rgba(106,63,255,0.20);
}

.fv-upload-title {
    margin-top: 20px;
    color: #ffffff;
    font-size: 25px;
    font-weight: 900;
    letter-spacing: -0.03em;
}

.fv-upload-description {
    margin-top: 10px;
    color: #7e8190;
    font-size: 13px;
    line-height: 1.7;
}

[data-testid="stFileUploader"] {
    margin-top: -12px;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius: 18px !important;
    border:
        1px solid rgba(255,255,255,0.07) !important;
    background:
        rgba(255,255,255,0.025) !important;
}

[data-testid="stFileUploaderFile"] {
    border-radius: 12px !important;
    background:
        rgba(255,255,255,0.035) !important;
    border:
        1px solid rgba(255,255,255,0.07) !important;
}


/* =========================================================
   SELECT
   ========================================================= */

[data-testid="stSelectbox"] label {
    color: #898c9d !important;
    font-size: 10px !important;
    font-weight: 900 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}


/* =========================================================
   ANALYSIS
   ========================================================= */

.fv-analysis {
    padding: 28px;
    border:
        1px solid rgba(255,255,255,0.075);
    border-radius: 24px;
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.045),
            rgba(255,255,255,0.012)
        );
    backdrop-filter: blur(18px);
}

.fv-analysis-label {
    color: #898b9c;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.fv-analysis-title {
    margin-top: 9px;
    color: white;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: -0.04em;
}

.fv-metric {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 15px 0;
    border-bottom:
        1px solid rgba(255,255,255,0.055);
}

.fv-metric-label {
    color: #acaebb;
    font-size: 13px;
}

.fv-metric-value {
    color: #6fffd0;
    font-size: 11px;
    font-weight: 900;
}


/* =========================================================
   STATUS
   ========================================================= */

.fv-status {
    padding: 17px 20px;
    border:
        1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    background:
        rgba(255,255,255,0.025);
    color: #aeb0bf;
    font-size: 13px;
    line-height: 1.65;
}

.fv-status strong {
    color: #6fffd0;
}


/* =========================================================
   RESULT CARDS
   ========================================================= */

[data-testid="stImage"] img {
    border-radius: 20px;
    box-shadow:
        0 25px 80px rgba(0,0,0,0.35);
    object-fit: contain;
}

.fv-result {
    padding: 10px;
    border:
        1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.015)
        );
}

.fv-rank {
    display: inline-flex;
    padding: 7px 10px;
    border-radius: 999px;
    background:
        rgba(8,9,13,0.78);
    border:
        1px solid rgba(255,255,255,0.11);
    color: white;
    font-size: 10px;
    font-weight: 900;
}

.fv-distance {
    display: inline-block;
    margin-top: 10px;
    padding: 6px 10px;
    border-radius: 999px;
    background:
        rgba(124,60,255,0.09);
    border:
        1px solid rgba(124,60,255,0.18);
    color: #b3a5ff;
    font-size: 10px;
    font-weight: 900;
}

.fv-resolution {
    display: inline-block;
    margin-top: 7px;
    color: #6fffd0;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}


/* =========================================================
   INFO
   ========================================================= */

.fv-info {
    margin-top: 12px;
    padding: 12px 14px;
    border-radius: 14px;
    border:
        1px solid rgba(255,255,255,0.07);
    background:
        rgba(255,255,255,0.025);
    color: #888b9c;
    font-size: 11px;
    line-height: 1.6;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    min-height: 48px;
    border-radius: 14px !important;
    border:
        1px solid rgba(255,255,255,0.10) !important;
    background:
        linear-gradient(
            135deg,
            rgba(124,60,255,0.22),
            rgba(0,212,255,0.10)
        ) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
}


/* =========================================================
   FOOTER
   ========================================================= */

.fv-footer {
    display: flex;
    justify-content: space-between;
    margin-top: 110px;
    padding-top: 26px;
    border-top:
        1px solid rgba(255,255,255,0.06);
    color: #646776;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 900px) {

    .fv-nav {
        margin-bottom: 50px;
    }

    .fv-hero-title {
        font-size:
            clamp(48px, 13vw, 90px);
    }

    .fv-hero-description {
        font-size: 15px;
    }

    .fv-footer {
        flex-direction: column;
        gap: 10px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# NAVIGATION
# =========================================================

render_html(
    """
<div class="fv-nav">

    <div class="fv-brand">

        <div class="fv-logo">
            ✦
        </div>

        <div class="fv-brand-text">
            Fashion Vision
        </div>

    </div>

    <div class="fv-online">
        <span class="fv-online-dot"></span>
        AI Online
    </div>

</div>
"""
)


# =========================================================
# HERO
# =========================================================

render_html(
    """
<div class="fv-hero">

    <div class="fv-eyebrow">
        COMPUTER VISION · VISUAL RETRIEVAL
    </div>

    <div class="fv-hero-title">
        DISCOVER<br>
        WHAT LOOKS<br>
        LIKE YOU.
    </div>

    <div class="fv-hero-description">
        Upload any fashion image and let a deep visual
        retrieval engine discover visually similar products
        from a catalog of more than 44,000 images.
    </div>

</div>
"""
)


# =========================================================
# LOAD FEATURE DATABASE
# =========================================================

@st.cache_data
def load_feature_database():

    with open(
        FEATURES_FILE,
        "rb",
    ) as file:
        features = pickle.load(file)

    with open(
        IMAGE_FILES_FILE,
        "rb",
    ) as file:
        image_files = pickle.load(file)

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if len(features) != len(image_files):
        raise ValueError(
            "The number of embeddings does not match "
            "the number of image paths."
        )

    return features, image_files


try:

    features_list, img_files_list = (
        load_feature_database()
    )

except Exception as exc:

    st.error(
        f"Could not load the feature database: {exc}"
    )

    st.stop()


catalog_size = len(features_list)


# =========================================================
# LOAD HIGH-RESOLUTION MAPPING
# =========================================================

@st.cache_data
def load_hf_mapping():

    mapping_path = Path(
        HF_MAPPING_FILE
    )

    if not mapping_path.exists():
        return {}

    try:

        with open(
            mapping_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return {}


hf_mapping = load_hf_mapping()


# =========================================================
# LOAD RESNET50
# =========================================================

@st.cache_resource
def load_model():

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False

    return Sequential(
        [
            base_model,
            GlobalMaxPooling2D(),
        ]
    )


try:

    model = load_model()

except Exception as exc:

    st.error(
        f"Could not load ResNet50: {exc}"
    )

    st.stop()


# =========================================================
# QUERY HEADER
# =========================================================

query_col, control_col = st.columns(
    [2.5, 1],
    gap="large",
)

with query_col:

    render_html(
        """
<div class="fv-section-eyebrow">
    01 · QUERY
</div>

<div class="fv-section-title">
    Upload your inspiration.
</div>

<div class="fv-section-description">
    Give the system an image and let visual similarity
    do the searching.
</div>
"""
    )

with control_col:

    top_k = st.selectbox(
        "RESULT COUNT",
        options=[5, 10, 15],
        index=0,
    )


# =========================================================
# MODE CARD
# =========================================================

render_html(
    f"""
<div class="fv-mode-card">

    Current mode:

    <span class="fv-mode-value">
        {APP_MODE_LABEL}
    </span>

    &nbsp;·&nbsp;

    {APP_MODE_DESCRIPTION}

    &nbsp;·&nbsp;

    Retrieval catalog:

    <span class="fv-mode-value">
        {catalog_size:,}
    </span>

    &nbsp;·&nbsp;

    High-res mappings:

    <span class="fv-mode-value">
        {len(hf_mapping):,}
    </span>

</div>
"""
)


# =========================================================
# UPLOAD AREA
# =========================================================

render_html(
    """
<div class="fv-upload">

    <div class="fv-upload-icon">
        ↑
    </div>

    <div class="fv-upload-title">
        Drop your image here
    </div>

    <div class="fv-upload-description">
        JPG · JPEG · PNG · WEBP
        <br>
        Visual search powered by ResNet50
    </div>

</div>
"""
)


uploaded_file = st.file_uploader(
    "Upload image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    label_visibility="collapsed",
)


# =========================================================
# NEAREST NEIGHBOR INDEX
# =========================================================

neighbors = NearestNeighbors(
    n_neighbors=min(
        50,
        catalog_size,
    ),
    algorithm="brute",
    metric="euclidean",
)

neighbors.fit(features_list)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_img_features(
    img,
    model,
):

    img = img.convert("RGB")

    img = img.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(
        img,
        dtype=np.float32,
    )

    batch = np.expand_dims(
        image_array,
        axis=0,
    )

    batch = preprocess_input(
        batch
    )

    result = model.predict(
        batch,
        verbose=0,
    )

    feature_vector = result.flatten()

    feature_norm = norm(
        feature_vector
    )

    if feature_norm == 0:
        return feature_vector

    return (
        feature_vector
        / feature_norm
    )


# =========================================================
# PRODUCT ID
# =========================================================

def extract_product_id(
    image_path,
):

    try:

        return int(
            Path(
                str(image_path)
            ).stem
        )

    except (
        ValueError,
        TypeError,
    ):

        return None


# =========================================================
# HUGGING FACE REQUEST
# =========================================================

def request_rows(
    row_index,
    retries=4,
):

    params = {
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "split": HF_SPLIT,
        "offset": int(row_index),
        "length": 1,
    }

    for attempt in range(
        retries + 1
    ):

        try:

            response = requests.get(
                HF_ROWS_API,
                params=params,
                headers=HF_HEADERS,
                timeout=45,
            )

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:

                    try:
                        delay = float(
                            retry_after
                        )
                    except ValueError:
                        delay = 2.0

                else:

                    delay = (
                        2.0
                        * (
                            2 ** attempt
                        )
                    )

                if attempt < retries:

                    time.sleep(
                        min(
                            delay,
                            15.0,
                        )
                    )

                    continue

                return None


            if (
                response.status_code >= 500
                and attempt < retries
            ):

                time.sleep(
                    min(
                        2.0
                        * (
                            2 ** attempt
                        ),
                        12.0,
                    )
                )

                continue


            if not response.ok:
                return None

            response.raise_for_status()

            return response.json()

        except requests.RequestException:

            if attempt < retries:

                time.sleep(
                    min(
                        2.0
                        * (
                            2 ** attempt
                        ),
                        12.0,
                    )
                )

                continue

            return None

    return None


# =========================================================
# HIGH-RES IMAGE
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_high_res_image(
    product_id,
):

    if product_id is None:
        return None, None, None

    product_info = hf_mapping.get(
        str(product_id)
    )

    if product_info is None:
        return None, None, None

    row_index = product_info.get(
        "hf_row_index"
    )

    if row_index is None:
        return None, None, None

    data = request_rows(
        row_index
    )

    if not data:
        return None, None, None

    rows = data.get(
        "rows",
        [],
    )

    if not rows:
        return None, None, None

    row = rows[0].get(
        "row",
        {},
    )

    # Verify the row belongs to requested product.
    returned_product_id = row.get(
        "id"
    )

    if returned_product_id is not None:

        try:

            if int(
                returned_product_id
            ) != int(
                product_id
            ):

                return None, None, None

        except (
            ValueError,
            TypeError,
        ):

            return None, None, None


    image_data = row.get(
        "image"
    )

    if not isinstance(
        image_data,
        dict,
    ):

        return None, None, None


    image_url = image_data.get(
        "src"
    )

    if not image_url:
        return None, None, None


    width = image_data.get(
        "width"
    )

    height = image_data.get(
        "height"
    )


    try:

        image_response = requests.get(
            image_url,
            headers=HF_HEADERS,
            timeout=45,
        )

        image_response.raise_for_status()

        return (
            image_response.content,
            width,
            height,
        )

    except requests.RequestException:

        return None, None, None


# =========================================================
# RESOLVE RECOMMENDATION
# =========================================================

def resolve_recommendation(
    image_path,
):

    product_id = extract_product_id(
        image_path
    )

    (
        image_bytes,
        width,
        height,
    ) = fetch_high_res_image(
        product_id
    )

    if image_bytes is None:

        return {
            "available": False,
            "product_id": product_id,
            "image": None,
            "width": None,
            "height": None,
        }

    return {
        "available": True,
        "product_id": product_id,
        "image": image_bytes,
        "width": width,
        "height": height,
    }


# =========================================================
# APPLICATION
# =========================================================

if uploaded_file is not None:

    try:

        # =================================================
        # QUERY IMAGE
        # =================================================

        query_image = Image.open(
            uploaded_file
        ).convert("RGB")


        # =================================================
        # ANALYSIS HEADER
        # =================================================

        st.markdown(
            "<div style='height:90px'></div>",
            unsafe_allow_html=True,
        )

        render_html(
            """
<div class="fv-section-eyebrow">
    02 · AI ANALYSIS
</div>

<div class="fv-section-title">
    Visual intelligence at work.
</div>

<div class="fv-section-description">
    Your image is converted into a deep visual embedding
    and searched against the full fashion catalog.
</div>
"""
        )

        st.markdown(
            "<div style='height:25px'></div>",
            unsafe_allow_html=True,
        )


        # =================================================
        # STATUS
        # =================================================

        status_box = st.empty()

        status_box.html(
            """
<div class="fv-status">

    <strong>01</strong>
    &nbsp;&nbsp;
    Reading image...

</div>
"""
        )


        # =================================================
        # EMBEDDING
        # =================================================

        features = extract_img_features(
            query_image,
            model,
        )


        status_box.html(
            """
<div class="fv-status">

    <strong>01 ✓</strong>
    &nbsp;&nbsp;
    Image loaded

    <br><br>

    <strong>02</strong>
    &nbsp;&nbsp;
    ResNet50 embedding generated...

</div>
"""
        )


        # =================================================
        # RETRIEVAL
        # =================================================

        candidate_count = min(
            max(
                top_k + 15,
                top_k * 4,
            ),
            catalog_size,
        )


        distances, indices = (
            neighbors.kneighbors(
                [features],
                n_neighbors=candidate_count,
            )
        )


        status_box.html(
            f"""
<div class="fv-status">

    <strong>01 ✓</strong>
    &nbsp;&nbsp;
    Image loaded

    <br><br>

    <strong>02 ✓</strong>
    &nbsp;&nbsp;
    ResNet50 embedding generated

    <br><br>

    <strong>03 ✓</strong>
    &nbsp;&nbsp;
    {catalog_size:,} products searched

    <br><br>

    <strong>04</strong>
    &nbsp;&nbsp;
    Resolving high-resolution images...

</div>
"""
        )


        # =================================================
        # RESOLVE HIGH-RES RESULTS
        # =================================================

        resolved_results = []
        failed_products = 0


        for distance, index in zip(
            distances[0],
            indices[0],
        ):

            image_path = (
                img_files_list[
                    int(index)
                ]
            )

            resolved = resolve_recommendation(
                image_path
            )

            if not resolved["available"]:

                failed_products += 1
                continue


            resolved_results.append(
                {
                    "distance": float(
                        distance
                    ),
                    "index": int(
                        index
                    ),
                    "product_id": (
                        resolved[
                            "product_id"
                        ]
                    ),
                    "image": (
                        resolved[
                            "image"
                        ]
                    ),
                    "width": (
                        resolved[
                            "width"
                        ]
                    ),
                    "height": (
                        resolved[
                            "height"
                        ]
                    ),
                }
            )


            if len(
                resolved_results
            ) >= top_k:

                break


        status_box.empty()


        # =================================================
        # QUERY + ANALYSIS
        # =================================================

        st.markdown(
            "<div style='height:35px'></div>",
            unsafe_allow_html=True,
        )


        image_col, analysis_col = st.columns(
            [1.15, 1],
            gap="large",
        )


        with image_col:

            st.image(
                query_image,
                width=500,
            )


        with analysis_col:

            render_html(
                f"""
<div class="fv-analysis">

    <div class="fv-analysis-label">
        VISION PIPELINE
    </div>

    <div class="fv-analysis-title">
        Query analyzed.
    </div>

    <div class="fv-metric">
        <div class="fv-metric-label">
            Backbone
        </div>

        <div class="fv-metric-value">
            ResNet50
        </div>
    </div>

    <div class="fv-metric">
        <div class="fv-metric-label">
            Pooling
        </div>

        <div class="fv-metric-value">
            Global Max
        </div>
    </div>

    <div class="fv-metric">
        <div class="fv-metric-label">
            Retrieval Catalog
        </div>

        <div class="fv-metric-value">
            {catalog_size:,}
        </div>
    </div>

    <div class="fv-metric">
        <div class="fv-metric-label">
            Search
        </div>

        <div class="fv-metric-value">
            Nearest Neighbor
        </div>
    </div>

    <div class="fv-metric">
        <div class="fv-metric-label">
            Distance
        </div>

        <div class="fv-metric-value">
            Euclidean
        </div>
    </div>

    <div class="fv-metric">
        <div class="fv-metric-label">
            High-Res Catalog
        </div>

        <div class="fv-metric-value">
            {len(hf_mapping):,}
        </div>
    </div>

</div>
"""
            )


        # =================================================
        # RESULTS HEADER
        # =================================================

        st.markdown(
            "<div style='height:95px'></div>",
            unsafe_allow_html=True,
        )

        render_html(
            f"""
<div class="fv-section-eyebrow">
    03 · RESULTS
</div>

<div class="fv-section-title">
    Top {top_k} visual matches.
</div>

<div class="fv-section-description">
    Results are ranked by the original 44,441-image
    retrieval system and displayed using the
    high-resolution catalog.
</div>
"""
        )

        st.markdown(
            "<div style='height:25px'></div>",
            unsafe_allow_html=True,
        )


        # =================================================
        # RESULTS GRID
        # =================================================

        if not resolved_results:

            st.error(
                "The retrieval engine found candidates, "
                "but no high-resolution catalog images "
                "could be loaded right now."
            )

            if failed_products > 0:

                st.caption(
                    f"{failed_products} candidate products "
                    "could not be resolved."
                )


        else:

            for start in range(
                0,
                len(resolved_results),
                5,
            ):

                row = resolved_results[
                    start:start + 5
                ]

                columns = st.columns(
                    len(row),
                    gap="medium",
                )


                for position, (
                    column,
                    result,
                ) in enumerate(
                    zip(
                        columns,
                        row,
                    )
                ):

                    rank = (
                        start
                        + position
                        + 1
                    )


                    with column:

                        render_html(
                            f"""
<div class="fv-result">

    <span class="fv-rank">
        #{rank:02d}
    </span>

</div>
"""
                        )


                        st.image(
                            result["image"],
                            width=220,
                        )


                        render_html(
                            f"""
<div class="fv-distance">
    DISTANCE ·
    {result["distance"]:.4f}
</div>

<div class="fv-resolution">
    HIGH RES ·
    {result["width"]}
    ×
    {result["height"]}
</div>
"""
                        )


            if failed_products > 0:

                render_html(
                    f"""
<div class="fv-info">

    {failed_products}
    nearby candidates were skipped because
    their high-resolution image was unavailable.
    The next available high-resolution matches
    were displayed.

</div>
"""
                )


        # =================================================
        # NEW SEARCH
        # =================================================

        st.markdown(
            "<div style='height:65px'></div>",
            unsafe_allow_html=True,
        )

        _, button_col, _ = st.columns(
            [1, 1, 1]
        )


        with button_col:

            if st.button(
                "↻  START NEW SEARCH",
                use_container_width=True,
            ):

                st.rerun()


        # =================================================
        # FOOTER
        # =================================================

        st.markdown(
            "<div style='height:70px'></div>",
            unsafe_allow_html=True,
        )

        render_html(
            f"""
<div class="fv-footer">

    <div>
        FASHION VISION · CONTENT-BASED RETRIEVAL
    </div>

    <div>
        RESNET50 · {catalog_size:,} PRODUCTS · FULL
    </div>

</div>
"""
        )


    except Exception as exc:

        st.error(
            f"Could not process the uploaded image: {exc}"
        )