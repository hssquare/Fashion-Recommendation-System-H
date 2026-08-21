import pickle

import numpy as np
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
# CUSTOM HTML RENDERER
# =========================================================

def render_html(html_content):
    """
    Render custom HTML using Streamlit's HTML renderer.
    """
    st.html(html_content)


# =========================================================
# FULL CATALOG CONFIGURATION
# =========================================================

FEATURES_FILE = "image_features_embedding.pkl"
IMAGE_FILES_FILE = "img_files.pkl"

APP_MODE_LABEL = "FULL"
APP_MODE_DESCRIPTION = "44,441-image local catalog"


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

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


    /* =====================================================
       STREAMLIT CLEANUP
       ===================================================== */

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


    /* =====================================================
       NAVIGATION
       ===================================================== */

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


    /* =====================================================
       HERO
       ===================================================== */

    .fv-hero {
        position: relative;

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

        color: white;

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


    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

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


    /* =====================================================
       MODE CARD
       ===================================================== */

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


    /* =====================================================
       UPLOAD CARD
       ===================================================== */

    .fv-upload {
        margin-top: 32px;

        padding: 65px 30px;

        text-align: center;

        border:
            1px dashed rgba(126, 88, 255, 0.62);

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

        transition:
            transform 220ms ease,
            border-color 220ms ease,
            box-shadow 220ms ease;
    }

    .fv-upload:hover {
        transform: translateY(-4px);

        border-color:
            rgba(145,110,255,0.85);

        box-shadow:
            0 25px 100px rgba(87,45,200,0.14);
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


    /* =====================================================
       STREAMLIT FILE UPLOADER
       ===================================================== */

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


    /* =====================================================
       SELECT BOX
       ===================================================== */

    [data-testid="stSelectbox"] label {
        color: #898c9d !important;

        font-size: 10px !important;

        font-weight: 900 !important;

        letter-spacing: 0.15em !important;

        text-transform: uppercase !important;
    }


    /* =====================================================
       ANALYSIS CARD
       ===================================================== */

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


    /* =====================================================
       QUERY IMAGE
       ===================================================== */

    [data-testid="stImage"] img {
        border-radius: 20px;

        box-shadow:
            0 25px 80px rgba(0,0,0,0.35);
    }


    /* =====================================================
       RESULT CARDS
       ===================================================== */

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

        transition:
            transform 220ms ease,
            border-color 220ms ease,
            box-shadow 220ms ease;
    }

    .fv-result:hover {
        transform: translateY(-8px);

        border-color:
            rgba(131,98,255,0.62);

        box-shadow:
            0 25px 90px rgba(0,0,0,0.45);
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

        letter-spacing: 0.04em;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

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


    /* =====================================================
       BUTTON
       ===================================================== */

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

        transition:
            transform 180ms ease,
            border-color 180ms ease,
            background 180ms ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        border-color:
            rgba(124,60,255,0.55) !important;

        background:
            linear-gradient(
                135deg,
                rgba(124,60,255,0.35),
                rgba(0,212,255,0.15)
            ) !important;
    }


    /* =====================================================
       RESPONSIVE
       ===================================================== */

    @media (max-width: 900px) {

        .fv-nav {
            margin-bottom: 50px;
        }

        .fv-hero-title {
            font-size: clamp(48px, 13vw, 90px);
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
# LOAD FULL FEATURE DATABASE
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


catalog_size = len(
    features_list
)


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
# FULL MODE CARD
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

    Catalog size:

    <span class="fv-mode-value">
        {catalog_size:,}
    </span>

</div>
"""
)


# =========================================================
# UPLOAD DESIGN
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
# BUILD NEAREST NEIGHBOR INDEX
# =========================================================

search_count = min(
    top_k + 1,
    catalog_size,
)

neighbors = NearestNeighbors(
    n_neighbors=search_count,
    algorithm="brute",
    metric="euclidean",
)

neighbors.fit(
    features_list
)


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
# RETRIEVAL
# =========================================================

def retrieve(features):

    distances, indices = (
        neighbors.kneighbors(
            [features]
        )
    )

    return distances[0], indices[0]


# =========================================================
# PROCESS UPLOADED IMAGE
# =========================================================

if uploaded_file is not None:

    try:

        # =================================================
        # LOAD QUERY
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
        # PROCESSING STATUS
        # =================================================

        status_box = st.empty()


        status_box.markdown(
            """
            <div style="
                padding:18px;
                border-radius:18px;
                border:1px solid rgba(255,255,255,0.08);
                background:rgba(255,255,255,0.025);
                color:#aeb0bf;
                font-size:13px;
            ">
                <b style="color:white;">01</b>
                &nbsp;&nbsp;
                Reading image...
            </div>
            """,
            unsafe_allow_html=True,
        )


        # =================================================
        # FEATURE EXTRACTION
        # =================================================

        features = extract_img_features(
            query_image,
            model,
        )


        status_box.markdown(
            f"""
            <div style="
                padding:18px;
                border-radius:18px;
                border:1px solid rgba(0,255,180,0.14);
                background:rgba(0,255,180,0.035);
                color:#6fffd0;
                font-size:13px;
            ">
                <b>01 ✓</b>
                &nbsp;&nbsp;
                Image loaded
                <br><br>
                <b>02 ✓</b>
                &nbsp;&nbsp;
                ResNet50 embedding generated
                <br><br>
                <b style="color:white;">03</b>
                &nbsp;&nbsp;
                Searching {catalog_size:,} products...
            </div>
            """,
            unsafe_allow_html=True,
        )


        # =================================================
        # RETRIEVAL
        # =================================================

        distances, indices = retrieve(
            features
        )


        status_box.markdown(
            f"""
            <div style="
                padding:18px;
                border-radius:18px;
                border:1px solid rgba(0,255,180,0.14);
                background:rgba(0,255,180,0.035);
                color:#6fffd0;
                font-size:13px;
            ">
                <b>01 ✓</b>
                &nbsp;&nbsp;
                Image loaded
                <br><br>
                <b>02 ✓</b>
                &nbsp;&nbsp;
                ResNet50 embedding generated
                <br><br>
                <b>03 ✓</b>
                &nbsp;&nbsp;
                {catalog_size:,} products searched
                <br><br>
                <b>04 ✓</b>
                &nbsp;&nbsp;
                Top-{top_k} matches ranked
            </div>
            """,
            unsafe_allow_html=True,
        )


        status_box.empty()


        # =================================================
        # QUERY + PIPELINE
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
            Catalog
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
            Mode
        </div>

        <div class="fv-metric-value">
            FULL
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
    Lower distance indicates a closer visual embedding match.
</div>
"""
        )


        st.markdown(
            "<div style='height:25px'></div>",
            unsafe_allow_html=True,
        )


        # =================================================
        # RESULTS
        # =================================================

        results = list(
            zip(
                distances,
                indices,
            )
        )


        for start in range(
            0,
            min(
                top_k,
                len(results),
            ),
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

                distance, index = result

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


                    image_path = (
                        img_files_list[index]
                    )


                    st.image(
                        image_path,
                        width=220,
                    )


                    render_html(
                        f"""
<div class="fv-distance">
    DISTANCE · {distance:.4f}
</div>
"""
                    )


        # =================================================
        # START NEW SEARCH
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