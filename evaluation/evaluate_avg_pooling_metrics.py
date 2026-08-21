from pathlib import Path
import csv
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset metadata
METADATA_PATH = Path(
    r"C:\D Drive Storage\Project\FRS HHHHHHHH\archive\styles.csv"
)

# Average-pooling embeddings
FEATURES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "avg_pool_embeddings.pkl"
)

IMAGE_FILES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "avg_pool_img_files.pkl"
)


# =========================================================
# LOAD METADATA
# =========================================================

def load_metadata(csv_path):
    """
    Load styles.csv while handling commas inside
    productDisplayName.
    """

    rows = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.reader(file)

        header = next(reader)

        for row in reader:

            if len(row) < 10:
                continue

            fixed_fields = row[:9]

            product_name = ",".join(
                row[9:]
            )

            rows.append(
                fixed_fields + [product_name]
            )

    return pd.DataFrame(
        rows,
        columns=header
    )


# =========================================================
# GET PRODUCT ID FROM IMAGE PATH
# =========================================================

def get_product_id(image_path):
    """
    Example:
        fashion_small/images/10000.jpg
        -> 10000
    """

    normalized_path = str(
        image_path
    ).replace("\\", "/")

    filename = Path(
        normalized_path
    ).name

    return Path(filename).stem


# =========================================================
# LOAD AVERAGE-POOLING EMBEDDINGS
# =========================================================

def load_feature_database():

    print("Loading average-pooling feature database...")

    with FEATURES_PATH.open(
        "rb"
    ) as file:

        features = pickle.load(file)

    with IMAGE_FILES_PATH.open(
        "rb"
    ) as file:

        image_files = pickle.load(file)

    features = np.asarray(
        features,
        dtype=np.float32
    )

    print(
        f"Embeddings: {len(features)}"
    )

    print(
        f"Image paths: {len(image_files)}"
    )

    return features, image_files


# =========================================================
# BUILD METADATA MAP
# =========================================================

def build_metadata_map(metadata):

    metadata["id"] = (
        metadata["id"]
        .astype(str)
        .str.strip()
    )

    return metadata.set_index(
        "id"
    ).to_dict(
        orient="index"
    )


# =========================================================
# PRECISION@K
# =========================================================

def precision_at_k(
    query_category,
    retrieved_categories,
    k
):

    retrieved = retrieved_categories[:k]

    if len(retrieved) == 0:
        return 0.0

    relevant = sum(
        category == query_category
        for category in retrieved
    )

    return relevant / len(retrieved)


# =========================================================
# RECALL@K
# =========================================================

def recall_at_k(
    query_category,
    retrieved_categories,
    total_relevant,
    k
):

    if total_relevant == 0:
        return 0.0

    retrieved = retrieved_categories[:k]

    relevant_retrieved = sum(
        category == query_category
        for category in retrieved
    )

    return (
        relevant_retrieved
        / total_relevant
    )


# =========================================================
# AVERAGE PRECISION FOR ONE QUERY
# =========================================================

def average_precision(
    query_category,
    retrieved_categories
):

    relevant_count = 0
    precision_sum = 0.0

    total_relevant = sum(
        category == query_category
        for category in retrieved_categories
    )

    if total_relevant == 0:
        return 0.0

    for rank, category in enumerate(
        retrieved_categories,
        start=1
    ):

        if category == query_category:

            relevant_count += 1

            precision_sum += (
                relevant_count / rank
            )

    return (
        precision_sum
        / total_relevant
    )


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():

    print(
        "=" * 60
    )

    print(
        "GLOBAL AVERAGE POOLING EVALUATION"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------------------
    # Load metadata
    # -----------------------------------------------------

    print(
        "\nLoading metadata..."
    )

    metadata = load_metadata(
        METADATA_PATH
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    # -----------------------------------------------------
    # Load embeddings
    # -----------------------------------------------------

    features, image_files = (
        load_feature_database()
    )

    # -----------------------------------------------------
    # Validate counts
    # -----------------------------------------------------

    if len(features) != len(image_files):

        raise ValueError(
            "Embedding count and image-path count do not match."
        )

    # -----------------------------------------------------
    # Build metadata lookup
    # -----------------------------------------------------

    metadata_map = (
        build_metadata_map(
            metadata
        )
    )

    # -----------------------------------------------------
    # Match embeddings to article types
    # -----------------------------------------------------

    valid_indices = []

    categories = []

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

        valid_indices.append(
            index
        )

        categories.append(
            str(article_type)
        )

    print(
        f"Valid evaluation images: "
        f"{len(valid_indices)}"
    )

    # -----------------------------------------------------
    # Filter features
    # -----------------------------------------------------

    evaluation_features = (
        features[valid_indices]
    )

    evaluation_image_files = [
        image_files[index]
        for index in valid_indices
    ]

    evaluation_categories = (
        categories
    )

    # -----------------------------------------------------
    # Build nearest-neighbor index
    # -----------------------------------------------------

    max_k = 15

    print(
        "\nBuilding nearest-neighbor index..."
    )

    neighbors = NearestNeighbors(
        n_neighbors=max_k + 1,
        algorithm="brute",
        metric="euclidean"
    )

    neighbors.fit(
        evaluation_features
    )

    # -----------------------------------------------------
    # Select deterministic sample
    # -----------------------------------------------------

    sample_size = min(
        500,
        len(evaluation_features)
    )

    rng = np.random.default_rng(
        seed=42
    )

    query_indices = rng.choice(
        len(evaluation_features),
        size=sample_size,
        replace=False
    )

    print(
        f"\nEvaluating {sample_size} queries..."
    )

    # -----------------------------------------------------
    # Metric storage
    # -----------------------------------------------------

    precision_5 = []
    precision_10 = []
    precision_15 = []

    recall_5 = []
    recall_10 = []
    recall_15 = []

    map_5 = []
    map_10 = []
    map_15 = []

    # -----------------------------------------------------
    # Evaluate each query
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

        # ---------------------------------------------
        # Find nearest neighbors
        # ---------------------------------------------

        distances, indices = (
            neighbors.kneighbors(
                [query_feature]
            )
        )

        # ---------------------------------------------
        # Remove query itself
        # ---------------------------------------------

        filtered_indices = []

        for index in indices[0]:

            if index != query_index:
                filtered_indices.append(
                    index
                )

            if len(filtered_indices) == max_k:
                break

        retrieved_categories = [
            evaluation_categories[index]
            for index in filtered_indices
        ]

        # ---------------------------------------------
        # Total relevant products
        # ---------------------------------------------

        total_relevant = sum(
            category == query_category
            for category in evaluation_categories
        ) - 1

        if total_relevant < 0:
            total_relevant = 0

        # ---------------------------------------------
        # Precision
        # ---------------------------------------------

        precision_5.append(
            precision_at_k(
                query_category,
                retrieved_categories,
                5
            )
        )

        precision_10.append(
            precision_at_k(
                query_category,
                retrieved_categories,
                10
            )
        )

        precision_15.append(
            precision_at_k(
                query_category,
                retrieved_categories,
                15
            )
        )

        # ---------------------------------------------
        # Recall
        # ---------------------------------------------

        recall_5.append(
            recall_at_k(
                query_category,
                retrieved_categories,
                total_relevant,
                5
            )
        )

        recall_10.append(
            recall_at_k(
                query_category,
                retrieved_categories,
                total_relevant,
                10
            )
        )

        recall_15.append(
            recall_at_k(
                query_category,
                retrieved_categories,
                total_relevant,
                15
            )
        )

        # ---------------------------------------------
        # mAP
        # ---------------------------------------------

        map_5.append(
            average_precision(
                query_category,
                retrieved_categories[:5]
            )
        )

        map_10.append(
            average_precision(
                query_category,
                retrieved_categories[:10]
            )
        )

        map_15.append(
            average_precision(
                query_category,
                retrieved_categories[:15]
            )
        )

        # ---------------------------------------------
        # Progress
        # ---------------------------------------------

        if count % 50 == 0:

            print(
                f"Processed "
                f"{count}/{sample_size}"
            )

    # =====================================================
    # RESULTS
    # =====================================================

    print()
    print(
        "=" * 60
    )

    print(
        "GLOBAL AVERAGE POOLING RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Queries evaluated : "
        f"{sample_size}"
    )

    print()

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

    print()

    print(
        f"Recall@5          : "
        f"{np.mean(recall_5):.4f}"
    )

    print(
        f"Recall@10         : "
        f"{np.mean(recall_10):.4f}"
    )

    print(
        f"Recall@15         : "
        f"{np.mean(recall_15):.4f}"
    )

    print()

    print(
        f"mAP@5             : "
        f"{np.mean(map_5):.4f}"
    )

    print(
        f"mAP@10            : "
        f"{np.mean(map_10):.4f}"
    )

    print(
        f"mAP@15            : "
        f"{np.mean(map_15):.4f}"
    )

    print(
        "=" * 60
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()