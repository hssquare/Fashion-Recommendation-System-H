from pathlib import Path
import csv
import pickle

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
    """
    Load the dataset metadata safely.

    The productDisplayName field may contain commas,
    so everything after the first 9 fields is treated
    as the product name.
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
# GET PRODUCT ID
# =========================================================

def get_product_id(image_path: str) -> str:
    """
    Example:
        fashion_small/images/1554.jpg
        -> 1554
    """

    normalized_path = str(
        image_path
    ).replace("\\", "/")

    filename = Path(
        normalized_path
    ).name

    return Path(filename).stem


# =========================================================
# LOAD FEATURE DATABASE
# =========================================================

def load_feature_database():

    with open(
        FEATURES_PATH,
        "rb"
    ) as file:

        features = pickle.load(file)

    with open(
        IMAGE_FILES_PATH,
        "rb"
    ) as file:

        image_files = pickle.load(file)

    features = np.asarray(
        features,
        dtype=np.float32
    )

    return features, image_files


# =========================================================
# BUILD METADATA MAP
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

    if total_relevant <= 0:
        return 0.0

    retrieved = retrieved_categories[:k]

    relevant_retrieved = sum(
        category == query_category
        for category in retrieved
    )

    return relevant_retrieved / total_relevant


# =========================================================
# AVERAGE PRECISION @ K
# =========================================================

def average_precision_at_k(
    query_category,
    retrieved_categories,
    total_relevant,
    k
):

    if total_relevant <= 0:
        return 0.0

    retrieved = retrieved_categories[:k]

    relevant_count = 0
    precision_sum = 0.0

    for rank, category in enumerate(
        retrieved,
        start=1
    ):

        if category == query_category:

            relevant_count += 1

            precision = (
                relevant_count / rank
            )

            precision_sum += precision

    # AP@K uses the smaller number of:
    # total relevant items available
    # or K retrieved positions.
    denominator = min(
        total_relevant,
        k
    )

    if denominator == 0:
        return 0.0

    return precision_sum / denominator


# =========================================================
# MAIN
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

    features, image_files = (
        load_feature_database()
    )

    print(
        f"Embeddings: {len(features)}"
    )

    print(
        f"Image paths: {len(image_files)}"
    )

    # -----------------------------------------------------
    # Validate counts
    # -----------------------------------------------------

    if len(features) != len(image_files):

        raise ValueError(
            "Feature count and image-path count do not match."
        )

    # -----------------------------------------------------
    # Metadata lookup
    # -----------------------------------------------------

    metadata_map = build_metadata_map(
        metadata
    )

    # -----------------------------------------------------
    # Match embeddings to metadata
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
            str(article_type).strip()
        )

    print(
        f"Valid evaluation images: "
        f"{len(valid_indices)}"
    )

    # -----------------------------------------------------
    # Filter embeddings
    # -----------------------------------------------------

    evaluation_features = features[
        valid_indices
    ]

    evaluation_categories = (
        product_categories
    )

    # -----------------------------------------------------
    # Build category counts
    # -----------------------------------------------------

    category_counts = {}

    for category in evaluation_categories:

        category_counts[category] = (
            category_counts.get(
                category,
                0
            ) + 1
        )

    # -----------------------------------------------------
    # Build retrieval model
    # -----------------------------------------------------

    max_k = 15

    neighbors = NearestNeighbors(
        n_neighbors=max_k + 1,
        algorithm="brute",
        metric="euclidean"
    )

    neighbors.fit(
        evaluation_features
    )

    # -----------------------------------------------------
    # Evaluation sample
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

    print(
        f"Evaluating {sample_size} queries..."
    )

    # =====================================================
    # QUERY LOOP
    # =====================================================

    for count, query_index in enumerate(
        query_indices,
        start=1
    ):

        query_category = (
            evaluation_categories[
                query_index
            ]
        )

        # -------------------------------------------------
        # Total relevant products
        #
        # We subtract 1 because the query image itself
        # belongs to the same articleType.
        # -------------------------------------------------

        total_relevant = (
            category_counts[
                query_category
            ] - 1
        )

        # -------------------------------------------------
        # Retrieve nearest neighbors
        # -------------------------------------------------

        query_feature = (
            evaluation_features[
                query_index
            ]
        )

        distances, indices = (
            neighbors.kneighbors(
                [query_feature]
            )
        )

        # -------------------------------------------------
        # Remove query image itself
        # -------------------------------------------------

        filtered_indices = [
            index
            for index in indices[0]
            if index != query_index
        ]

        retrieved_categories = [
            evaluation_categories[index]
            for index in filtered_indices
        ]

        # -------------------------------------------------
        # Precision
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Recall
        # -------------------------------------------------

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

        # -------------------------------------------------
        # AP / mAP
        # -------------------------------------------------

        map_5.append(
            average_precision_at_k(
                query_category,
                retrieved_categories,
                total_relevant,
                5
            )
        )

        map_10.append(
            average_precision_at_k(
                query_category,
                retrieved_categories,
                total_relevant,
                10
            )
        )

        map_15.append(
            average_precision_at_k(
                query_category,
                retrieved_categories,
                total_relevant,
                15
            )
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

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
        "=" * 65
    )

    print(
        "CONTENT-BASED IMAGE RETRIEVAL EVALUATION"
    )

    print(
        "=" * 65
    )

    print(
        f"Queries evaluated : {sample_size}"
    )

    print()

    print(
        "Precision@5       : "
        f"{np.mean(precision_5):.4f}"
    )

    print(
        "Recall@5          : "
        f"{np.mean(recall_5):.4f}"
    )

    print(
        "mAP@5             : "
        f"{np.mean(map_5):.4f}"
    )

    print()

    print(
        "Precision@10      : "
        f"{np.mean(precision_10):.4f}"
    )

    print(
        "Recall@10         : "
        f"{np.mean(recall_10):.4f}"
    )

    print(
        "mAP@10            : "
        f"{np.mean(map_10):.4f}"
    )

    print()

    print(
        "Precision@15      : "
        f"{np.mean(precision_15):.4f}"
    )

    print(
        "Recall@15         : "
        f"{np.mean(recall_15):.4f}"
    )

    print(
        "mAP@15            : "
        f"{np.mean(map_15):.4f}"
    )

    print()

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()