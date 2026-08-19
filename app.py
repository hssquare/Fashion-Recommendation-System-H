from pathlib import Path
import pickle

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential


# Dataset location
DATASET_DIR = Path("fashion_small") / "images"

# Validate dataset directory
if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR.resolve()}"
    )

# Load pretrained ResNet50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the pretrained model
base_model.trainable = False

# Add global max pooling
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])


def extract_features(img_path, model):
    """Extract and L2-normalize a ResNet50 feature vector."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img, verbose=0)

    flattened_result = result.flatten()

    feature_norm = norm(flattened_result)

    if feature_norm == 0:
        return flattened_result

    normalized_result = flattened_result / feature_norm

    return normalized_result


# Collect image paths
img_files = []

for fashion_image in DATASET_DIR.iterdir():
    if fashion_image.is_file():
        img_files.append(str(fashion_image))

# Sort for reproducible ordering
img_files.sort()

print(f"Found {len(img_files)} images.")

# Extract image features
image_features = []

for img_path in tqdm(img_files, desc="Extracting features"):
    try:
        features = extract_features(img_path, model)
        image_features.append(features)
    except Exception as exc:
        print(f"\nSkipping {img_path}: {exc}")

# Save feature database
with open("image_features_embedding.pkl", "wb") as f:
    pickle.dump(image_features, f)

with open("img_files.pkl", "wb") as f:
    pickle.dump(img_files[:len(image_features)], f)

print("\nFeature extraction completed.")
print(f"Embeddings saved: {len(image_features)}")
print("Saved: image_features_embedding.pkl")
print("Saved: img_files.pkl")