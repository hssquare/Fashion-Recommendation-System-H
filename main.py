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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fashion Vision",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PROJECT CONFIG
# =========================================================

FEATURES_FILE = "image_features_embedding.pkl"
IMAGE_FILES_FILE = "img_files.pkl"
HF_MAPPING_FILE = "huggingface_product_mapping.json"

# YOUR OWN PUBLIC HIGH-RES DATASET
HF_DATASET = "GangHitman/fashion-recommendation-images"
HF_CONFIG = "default"
HF_SPLIT = "train"

HF_API_BASE = "https://datasets-server.huggingface.co"

HF_ROWS_API = f"{HF_API_BASE}/rows"
HF_VALID_API = f"{HF_API_BASE}/is-valid"
HF_FIRST_ROWS_API = f"{HF_API_BASE}/first-rows"

APP_MODE_LABEL = "FULL"
APP_MODE_DESCRIPTION = "44,441-image retrieval catalog"

# Dataset Viewer permits up to 100 rows per /rows request.
HF_BLOCK_SIZE = 100

# Number of visual candidates to inspect before
# resolving high-resolution images.
EXTRA_CANDIDATES = 25


# =========================================================
# HUGGING FACE SECRET
# =========================================================

try:
    HF_TOKEN = st.secrets.get(
        "HF_TOKEN",
        "",
    ).strip()
except Exception:
    HF_TOKEN = ""


# Authenticated headers.
HF_AUTH_HEADERS = {
    "User-Agent": "FashionVision/1.0",
}

if HF_TOKEN:
    HF_AUTH_HEADERS["Authorization"] = (
        f"Bearer {HF_TOKEN}"
    )


# Public headers.
# Your duplicated dataset is public, so this gives us
# a second access path if authentication is unavailable.
HF_PUBLIC_HEADERS = {
    "User-Agent": "FashionVision/1.0",
}


# =========================================================
# HTML HELPER
# =========================================================

def html(body):
    st.html(body)


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
   NAV
   ========================================================= */

.fv-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 18px 22px;
    margin-bottom: 70px;

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

    color: white;
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
    padding: 10px 0 85px;
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
   SECTION
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
   MODE
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
}

.fv-upload-title {
    margin-top: 20px;

    color: white;

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
   RESULTS
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
   DEBUG
   ========================================================= */

.fv-debug {
    margin-top: 12px;

    padding: 14px 16px;

    border:
        1px solid rgba(255,95,95,0.18);

    border-radius: 14px;

    background:
        rgba(255,70,70,0.05);

    color: #ff9999;

    font-size: 11px;

    line-height: 1.6;

    word-break: break-word;
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

    color: white !important;

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
   MOBILE
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

html(
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

html(
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
# FEATURE DATABASE
# =========================================================

@st.cache_data
def load_feature_database():

    feature_path = Path(
        FEATURES_FILE
    )

    image_path = Path(
        IMAGE_FILES_FILE
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_FILE}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing {IMAGE_FILES_FILE}"
        )

    with open(
        feature_path,
        "rb",
    ) as file:

        features = pickle.load(file)

    with open(
        image_path,
        "rb",
    ) as file:

        image_files = pickle.load(file)

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if len(features) != len(
        image_files
    ):

        raise ValueError(
            "Embedding count does not match "
            "image-path count."
        )

    return (
        features,
        image_files,
    )


try:

    features_list, img_files_list = (
        load_feature_database()
    )

except Exception as exc:

    st.error(
        f"Could not load feature database: {exc}"
    )

    st.stop()


catalog_size = len(
    features_list
)


# =========================================================
# HIGH-RES MAPPING
# =========================================================

@st.cache_data
def load_hf_mapping():

    path = Path(
        HF_MAPPING_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing {HF_MAPPING_FILE}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        mapping = json.load(file)

    if not isinstance(
        mapping,
        dict,
    ):

        raise ValueError(
            "HF mapping is not a JSON object."
        )

    return mapping


try:

    hf_mapping = load_hf_mapping()

except Exception as exc:

    st.error(
        f"Could not load high-resolution mapping: {exc}"
    )

    st.stop()


# =========================================================
# RESNET50
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
# QUERY CONTROLS
# =========================================================

query_col, control_col = st.columns(
    [2.5, 1],
    gap="large",
)


with query_col:

    html(
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
# MODE INFO
# =========================================================

html(
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
# UPLOAD UI
# =========================================================

html(
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
# NEAREST NEIGHBOR
# =========================================================

neighbors = NearestNeighbors(
    n_neighbors=min(
        100,
        catalog_size,
    ),
    algorithm="brute",
    metric="euclidean",
)

neighbors.fit(
    features_list
)


# =========================================================
# IMAGE FEATURE EXTRACTION
# =========================================================

def extract_img_features(
    image,
):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    batch = np.expand_dims(
        array,
        axis=0,
    )

    batch = preprocess_input(
        batch
    )

    result = model.predict(
        batch,
        verbose=0,
    )

    vector = result.flatten()

    vector_norm = norm(
        vector
    )

    if vector_norm == 0:

        raise ValueError(
            "Generated embedding has zero norm."
        )

    return (
        vector
        / vector_norm
    )


# =========================================================
# PRODUCT ID FROM LOCAL FILENAME
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
# GENERIC REQUEST
# =========================================================

def get_hf_json(
    url,
    params,
    timeout=60,
):

    errors = []

    # Try authentication first when configured.
    headers_to_try = []

    if HF_TOKEN:

        headers_to_try.append(
            (
                "authenticated",
                HF_AUTH_HEADERS,
            )
        )

    # Then public access.
    headers_to_try.append(
        (
            "public",
            HF_PUBLIC_HEADERS,
        )
    )


    for header_name, headers in (
        headers_to_try
    ):

        for attempt in range(3):

            try:

                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )


                if response.ok:

                    return (
                        response.json(),
                        None,
                        header_name,
                    )


                # Retry only transient errors.
                if response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):

                    time.sleep(
                        min(
                            2 ** attempt,
                            8,
                        )
                    )

                    continue


                errors.append(
                    (
                        f"{header_name}: "
                        f"HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                )

                break


            except requests.RequestException as exc:

                errors.append(
                    (
                        f"{header_name}: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )
                )

                if attempt < 2:

                    time.sleep(
                        min(
                            2 ** attempt,
                            8,
                        )
                    )


    return (
        None,
        " | ".join(errors),
        None,
    )


# =========================================================
# HUGGING FACE DATASET PREFLIGHT
# =========================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def check_hf_dataset():

    params = {
        "dataset": HF_DATASET,
    }

    (
        data,
        error,
        auth_mode,
    ) = get_hf_json(
        HF_VALID_API,
        params,
    )

    if data is not None:

        return {
            "ok": True,
            "data": data,
            "error": None,
            "auth_mode": auth_mode,
        }

    return {
        "ok": False,
        "data": None,
        "error": error,
        "auth_mode": None,
    }


# =========================================================
# FETCH ROW BLOCK
#
# One call fetches up to 100 rows.
# This dramatically reduces the number of HF API requests.
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_row_block(
    block_start,
):

    length = HF_BLOCK_SIZE

    # Do not request beyond the dataset.
    if block_start + length > 44239:

        length = 44239 - block_start

    if length <= 0:

        return {
            "ok": False,
            "rows": {},
            "error": "Invalid row block.",
            "auth_mode": None,
        }


    params = {
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "split": HF_SPLIT,
        "offset": int(block_start),
        "length": int(length),
    }


    (
        data,
        error,
        auth_mode,
    ) = get_hf_json(
        HF_ROWS_API,
        params,
    )


    if data is None:

        return {
            "ok": False,
            "rows": {},
            "error": error,
            "auth_mode": None,
        }


    rows = {}

    for item in data.get(
        "rows",
        [],
    ):

        row_index = item.get(
            "row_idx"
        )

        row_data = item.get(
            "row"
        )


        if (
            row_index is not None
            and isinstance(
                row_data,
                dict,
            )
        ):

            rows[int(row_index)] = (
                row_data
            )


    return {
        "ok": True,
        "rows": rows,
        "error": None,
        "auth_mode": auth_mode,
    }


# =========================================================
# DOWNLOAD HIGH-RES IMAGE
# =========================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def download_high_res_image(
    image_url,
):

    if not image_url:

        return {
            "ok": False,
            "bytes": None,
            "error": "Image URL is empty.",
        }


    headers_to_try = []

    if HF_TOKEN:

        headers_to_try.append(
            HF_AUTH_HEADERS
        )

    headers_to_try.append(
        HF_PUBLIC_HEADERS
    )


    errors = []


    for headers in headers_to_try:

        for attempt in range(3):

            try:

                response = requests.get(
                    image_url,
                    headers=headers,
                    timeout=60,
                )


                if response.ok:

                    return {
                        "ok": True,
                        "bytes": response.content,
                        "error": None,
                    }


                if response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):

                    time.sleep(
                        min(
                            2 ** attempt,
                            8,
                        )
                    )

                    continue


                errors.append(
                    f"HTTP {response.status_code}"
                )

                break


            except requests.RequestException as exc:

                errors.append(
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt < 2:

                    time.sleep(
                        min(
                            2 ** attempt,
                            8,
                        )
                    )


    return {
        "ok": False,
        "bytes": None,
        "error": " | ".join(errors),
    }


# =========================================================
# HIGH-RES PRODUCT RESOLUTION
# =========================================================

def resolve_high_res_product(
    product_id,
):

    if product_id is None:

        return {
            "ok": False,
            "error": "No local product ID.",
        }


    mapping = hf_mapping.get(
        str(product_id)
    )


    if mapping is None:

        return {
            "ok": False,
            "error": (
                f"Product {product_id} "
                "is not present in HF mapping."
            ),
        }


    row_index = mapping.get(
        "hf_row_index"
    )


    if row_index is None:

        return {
            "ok": False,
            "error": (
                f"Product {product_id} "
                "has no hf_row_index."
            ),
        }


    try:

        row_index = int(
            row_index
        )

    except (
        TypeError,
        ValueError,
    ):

        return {
            "ok": False,
            "error": (
                f"Invalid row index for "
                f"product {product_id}."
            ),
        }


    block_start = (
        row_index
        // HF_BLOCK_SIZE
    ) * HF_BLOCK_SIZE


    block = fetch_row_block(
        block_start
    )


    if not block["ok"]:

        return {
            "ok": False,
            "error": (
                f"Dataset Viewer request failed "
                f"for row {row_index}: "
                f"{block['error']}"
            ),
        }


    row = block[
        "rows"
    ].get(
        row_index
    )


    if row is None:

        return {
            "ok": False,
            "error": (
                f"Row {row_index} was not returned "
                "by Hugging Face."
            ),
        }


    # -----------------------------------------------------
    # Verify dataset product ID.
    # -----------------------------------------------------

    returned_id = row.get(
        "id"
    )


    if returned_id is not None:

        try:

            if int(
                returned_id
            ) != int(
                mapping.get(
                    "hf_product_id",
                    returned_id,
                )
            ):

                # We don't fail only because the mapping
                # may not contain hf_product_id.
                # The actual row is still inspected below.

                pass

        except (
            ValueError,
            TypeError,
        ):

            pass


    # -----------------------------------------------------
    # Extract image.
    # -----------------------------------------------------

    image_data = row.get(
        "image"
    )


    if not isinstance(
        image_data,
        dict,
    ):

        return {
            "ok": False,
            "error": (
                f"Product {product_id} row "
                "does not contain an image object."
            ),
        }


    image_url = (
        image_data.get("src")
        or image_data.get("url")
    )


    if not image_url:

        return {
            "ok": False,
            "error": (
                f"Product {product_id} "
                "has no image.src URL."
            ),
        }


    width = image_data.get(
        "width"
    )

    height = image_data.get(
        "height"
    )


    # -----------------------------------------------------
    # Download actual image.
    # -----------------------------------------------------

    image_result = (
        download_high_res_image(
            image_url
        )
    )


    if not image_result["ok"]:

        return {
            "ok": False,
            "error": (
                f"Product {product_id} image "
                f"download failed: "
                f"{image_result['error']}"
            ),
        }


    return {
        "ok": True,
        "image": image_result["bytes"],
        "width": width,
        "height": height,
        "row_index": row_index,
        "hf_product_id": returned_id,
        "image_url": image_url,
    }


# =========================================================
# LOCAL RETRIEVAL
# =========================================================

def retrieve_candidates(
    query_vector,
    count,
):

    count = min(
        count,
        catalog_size,
    )


    distances, indices = (
        neighbors.kneighbors(
            [query_vector],
            n_neighbors=count,
        )
    )


    return [
        {
            "distance": float(distance),
            "index": int(index),
        }

        for distance, index in zip(
            distances[0],
            indices[0],
        )
    ]


# =========================================================
# MAIN PROCESSING
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
            "<div style='height:85px'></div>",
            unsafe_allow_html=True,
        )


        html(
            """
<div class="fv-section-eyebrow">
    02 · AI ANALYSIS
</div>

<div class="fv-section-title">
    Visual intelligence at work.
</div>

<div class="fv-section-description">
    Your image is converted into a deep visual embedding
    and searched against the selected catalog.
</div>
"""
        )


        st.markdown(
            "<div style='height:24px'></div>",
            unsafe_allow_html=True,
        )


        # =================================================
        # TEMPORARY STATUS
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

        query_vector = (
            extract_img_features(
                query_image
            )
        )


        status_box.html(
            """
<div class="fv-status">

    <strong>01 ✓</strong>
    &nbsp;&nbsp;
    Image loaded

    <br><br>

    <strong>02 ✓</strong>
    &nbsp;&nbsp;
    ResNet50 embedding generated

</div>
"""
        )


        # =================================================
        # LOCAL RETRIEVAL
        # =================================================

        candidate_count = min(
            max(
                top_k + EXTRA_CANDIDATES,
                top_k * 5,
            ),
            catalog_size,
        )


        candidates = retrieve_candidates(
            query_vector,
            candidate_count,
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
    Checking high-resolution catalog...

</div>
"""
        )


        # =================================================
        # HUGGING FACE PREFLIGHT
        # =================================================

        hf_status = (
            check_hf_dataset()
        )


        if not hf_status["ok"]:

            status_box.empty()


            st.error(
                "The visual retrieval model is working, "
                "but the Hugging Face Dataset Viewer "
                "is currently unavailable."
            )


            html(
                f"""
<div class="fv-debug">

    <strong>Hugging Face diagnostic</strong>

    <br><br>

    Dataset:
    {HF_DATASET}

    <br>

    Config:
    {HF_CONFIG}

    <br>

    Split:
    {HF_SPLIT}

    <br><br>

    {hf_status["error"]}

</div>
"""
            )

            st.stop()


        # =================================================
        # RESOLVE HIGH-RES RESULTS
        # =================================================

        results = []

        failed = []

        blocks_loaded = set()


        for candidate in candidates:

            local_index = candidate[
                "index"
            ]

            image_path = (
                img_files_list[
                    local_index
                ]
            )


            product_id = (
                extract_product_id(
                    image_path
                )
            )


            if product_id is None:

                failed.append(
                    {
                        "product_id": None,
                        "error": (
                            f"Could not extract "
                            f"product ID from "
                            f"{image_path}"
                        ),
                    }
                )

                continue


            # Load/cache the required block.
            mapping = hf_mapping.get(
                str(product_id)
            )


            if mapping:

                row_idx = mapping.get(
                    "hf_row_index"
                )

                if row_idx is not None:

                    block_start = (
                        int(row_idx)
                        // HF_BLOCK_SIZE
                    ) * HF_BLOCK_SIZE

                    if (
                        block_start
                        not in blocks_loaded
                    ):

                        blocks_loaded.add(
                            block_start
                        )


            resolved = (
                resolve_high_res_product(
                    product_id
                )
            )


            if not resolved["ok"]:

                failed.append(
                    {
                        "product_id": product_id,
                        "error": resolved[
                            "error"
                        ],
                    }
                )

                continue


            results.append(
                {
                    "distance": candidate[
                        "distance"
                    ],
                    "local_index": local_index,
                    "product_id": product_id,
                    "image": resolved[
                        "image"
                    ],
                    "width": resolved[
                        "width"
                    ],
                    "height": resolved[
                        "height"
                    ],
                    "row_index": resolved[
                        "row_index"
                    ],
                }
            )


            if len(results) >= top_k:

                break


        status_box.empty()


        # =================================================
        # IF NOTHING RESOLVED
        # =================================================

        if not results:

            st.error(
                "The 44,441-image retrieval engine "
                "worked, but no high-resolution images "
                "could be resolved."
            )


            if failed:

                first_error = failed[0]

                html(
                    f"""
<div class="fv-debug">

    <strong>First high-resolution lookup failure</strong>

    <br><br>

    Product:
    {first_error["product_id"]}

    <br><br>

    {first_error["error"]}

</div>
"""
                )


                with st.expander(
                    "Show all lookup failures"
                ):

                    for failure in failed[:20]:

                        st.code(
                            str(
                                failure
                            )
                        )


            st.stop()


        # =================================================
        # QUERY + PIPELINE
        # =================================================

        st.markdown(
            "<div style='height:30px'></div>",
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

            html(
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
            "<div style='height:85px'></div>",
            unsafe_allow_html=True,
        )


        html(
            f"""
<div class="fv-section-eyebrow">
    03 · RESULTS
</div>

<div class="fv-section-title">
    Top {len(results)} visual matches.
</div>

<div class="fv-section-description">
    Ranking comes from the original 44,441-image
    retrieval system. Images are displayed from
    your high-resolution Hugging Face catalog.
</div>
"""
        )


        st.markdown(
            "<div style='height:25px'></div>",
            unsafe_allow_html=True,
        )


        # =================================================
        # RESULT GRID
        # =================================================

        for start in range(
            0,
            len(results),
            5,
        ):

            row = results[
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

                    html(
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


                    html(
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


        # =================================================
        # SKIPPED ITEMS
        # =================================================

        if failed:

            html(
                f"""
<div class="fv-info">

    {len(failed)}
    nearby candidates were skipped because
    their high-resolution image could not be
    resolved. The next available matches
    were displayed.

</div>
"""
            )


            with st.expander(
                "Diagnostics"
            ):

                for failure in failed[:20]:

                    st.write(
                        failure
                    )


        # =================================================
        # NEW SEARCH
        # =================================================

        st.markdown(
            "<div style='height:60px'></div>",
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
            "<div style='height:65px'></div>",
            unsafe_allow_html=True,
        )


        html(
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
            "The application encountered an error."
        )


        # Never expose the HF token.
        safe_error = str(
            exc
        ).replace(
            HF_TOKEN,
            "[HF_TOKEN_REDACTED]"
        )


        html(
            f"""
<div class="fv-debug">

    <strong>Application diagnostic</strong>

    <br><br>

    {safe_error}

</div>
"""
        )