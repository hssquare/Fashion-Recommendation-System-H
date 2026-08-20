from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = Path(
    r"C:\D Drive Storage\Project\FRS HHHHHHHH\archive\styles.csv"
)

FEATURES_PATH = PROJECT_ROOT / "image_features_embedding.pkl"
IMAGE_FILES_PATH = PROJECT_ROOT / "img_files.pkl"


# =========================================================
# LOAD METADATA
# =========================================================

def load_metadata(csv_path: Path) -> pd.DataFrame:
    rows = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        import csv

        reader = csv.reader(file)
        header = next(reader)

        for row in reader:

            if len(row) < 10:
                continue

            fixed_fields = row[:9]
            product_name = ",".join(row[9:])

            rows.append(
                fixed_fields + [product_name]
            )

    df = pd.DataFrame(
        rows,
        columns=header
    )

    df["id"] = (
        df["id"]
        .astype(str)
        .str.strip()
    )

    return df


# =========================================================
# PRODUCT ID FROM IMAGE PATH
# =========================================================

def get_product_id(image_path: str) -> str:
    """
    Example:
        fashion_small/images/1554.jpg
        -> 1554
    """

    filename = Path(
        image_path.replace("\\", "/")
    ).name

    return Path(filename).stem


# =========================================================
# LOAD FEATURE DATABASE
# =========================================================

def load_feature_database():

    with open(FEATURES_PATH, "rb") as file:
        features = pickle.load(file)

    with open(IMAGE_FILES_PATH, "rb") as file:
        image_files = pickle.load(file)

    features = np.asarray(
        features,
        dtype=np.float32
    )

    return features, image_files


# =========================================================
# BUILD PRODUCT METADATA MAP
# =========================================================

def build_metadata_map(metadata):

    return metadata.set_index(
        "id"
    ).to_dict(
        orient="index"
    )


# =========================================================
# PRECISION@K
# =========================================================

def calculate_precision_at_k(
    query_category,
    retrieved_categories,
    k
):

    retrieved = retrieved_categories[:k]

    if not retrieved:
        return 0.0

    relevant = sum(
        category == query_category
        for category in retrieved
    )

    return relevant / len(retrieved)


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():

    print("Loading metadata...")

    metadata = load_metadata(
        METADATA_PATH
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    print("Loading feature database...")

    features, image_files = load_feature_database()

    print(
        f"Embeddings: {len(features)}"
    )

    print(
        f"Image paths: {len(image_files)}"
    )

    # -----------------------------------------------------
    # Make sure the counts match
    # -----------------------------------------------------

    if len(features) != len(image_files):

        raise ValueError(
            "Feature count and image-path count do not match."
        )

    # -----------------------------------------------------
    # Metadata dictionary
    # -----------------------------------------------------

    metadata_map = build_metadata_map(
        metadata
    )

    # -----------------------------------------------------
    # Map every embedding to metadata
    # -----------------------------------------------------

    valid_indices = []

    product_categories = []

    for index, image_path in enumerate(
        image_files
    ):

        product_id = get_product_id(
            image_path
        )

        record = metadata_map.get(
            product_id
        )

        if record is None:
            continue

        article_type = record.get(
            "articleType"
        )

        if pd.isna(article_type):
            continue

        valid_indices.append(index)
        product_categories.append(
            article_type
        )

    print(
        f"Valid evaluation images: "
        f"{len(valid_indices)}"
    )

    # -----------------------------------------------------
    # Filter feature matrix
    # -----------------------------------------------------

    evaluation_features = features[
        valid_indices
    ]

    evaluation_image_files = [
        image_files[index]
        for index in valid_indices
    ]

    evaluation_categories = (
        product_categories
    )

    # -----------------------------------------------------
    # Build nearest-neighbor model
    # -----------------------------------------------------

    max_k = 15

    neighbors = NearestNeighbors(
        n_neighbors=max_k + 1,
        algorithm="brute",
        metric="cosine"
    )

    neighbors.fit(
        evaluation_features
    )

    # -----------------------------------------------------
    # Evaluate subset
    # -----------------------------------------------------

    sample_size = min(
        500,
        len(evaluation_features)
    )

    # Deterministic sampling
    rng = np.random.default_rng(
        seed=42
    )

    query_indices = rng.choice(
        len(evaluation_features),
        size=sample_size,
        replace=False
    )

    precision_5 = []
    precision_10 = []
    precision_15 = []

    print(
        f"Evaluating {sample_size} queries..."
    )

    # -----------------------------------------------------
    # Query loop
    # -----------------------------------------------------

    for count, query_index in enumerate(
        query_indices,
        start=1
    ):

        query_feature = (
            evaluation_features[
                query_index
            ]
        )

        query_category = (
            evaluation_categories[
                query_index
            ]
        )

        distances, indices = (
            neighbors.kneighbors(
                [query_feature]
            )
        )

        # Remove the query image itself
        filtered_indices = [
            index
            for index in indices[0]
            if index != query_index
        ]

        retrieved_categories = [
            evaluation_categories[index]
            for index in filtered_indices
        ]

        precision_5.append(
            calculate_precision_at_k(
                query_category,
                retrieved_categories,
                5
            )
        )

        precision_10.append(
            calculate_precision_at_k(
                query_category,
                retrieved_categories,
                10
            )
        )

        precision_15.append(
            calculate_precision_at_k(
                query_category,
                retrieved_categories,
                15
            )
        )

        if count % 50 == 0:
            print(
                f"Processed {count}/{sample_size}"
            )

    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)

    print(
        f"Queries evaluated : {sample_size}"
    )

    print(
        f"Precision@5       : "
        f"{np.mean(precision_5):.4f}"
    )

    print(
        f"Precision@10      : "
        f"{np.mean(precision_10):.4f}"
    )

    print(
        f"Precision@15      : "
        f"{np.mean(precision_15):.4f}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()