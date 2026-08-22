# 🖼️ Fashion Recommendation System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![ResNet50](https://img.shields.io/badge/Model-ResNet50-8E44AD)](https://keras.io/api/applications/resnet/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Retrieval-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-3776AB)](https://python-pillow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-High--Res%20Catalog-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/hssquare/Fashion-Recommendation-System-H)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Large%20Files-82C91E)](https://git-lfs.com/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Fashion%20Vision-00C853?logo=streamlit&logoColor=white)](https://fashion-recommendation-system-h8w48qvsj6zbbnwyrtmk.streamlit.app/)
[![Release](https://img.shields.io/badge/Release-v1.0.0-00C853)](https://github.com/hssquare/Fashion-Recommendation-System-H/releases/tag/v1.0.0)

> 🧠 A **Content-Based Image Retrieval (CBIR)** system that uses deep visual features to find visually similar fashion products from a large product catalog.

Users upload a fashion or product image, and the system extracts its visual representation using a pretrained **ResNet50** model and retrieves the most similar catalog images using nearest-neighbor search.

---

## 📌 Overview

Traditional recommendation systems often rely on ratings, purchase history, user preferences, or collaborative filtering.

This project instead uses the **image itself as the query**.

The uploaded image is converted into a visual feature embedding using a pretrained **ResNet50 CNN**. The embedding is L2-normalized and compared with precomputed catalog embeddings to retrieve visually similar products.

The application is built with **Streamlit**.

---

## 🎯 Project Objectives

- 📷 Accept a fashion/product image as input
- 🧠 Extract deep visual features using ResNet50
- 🗂️ Build a searchable feature database
- 🔍 Retrieve visually similar products
- 📊 Rank results using nearest-neighbor search
- 🔢 Support Top 5, Top 10, and Top 15 retrieval
- 📏 Display retrieval distance
- 📈 Evaluate retrieval quality quantitatively
- 🌐 Deploy the application using Streamlit

---

# ⚙️ Methodology

## 1. Feature Extraction

The project uses **ImageNet-pretrained ResNet50 as a frozen feature extractor**.

The classification head is removed and **Global Max Pooling** is applied to generate a compact visual representation.

The model is **not trained or fine-tuned on the fashion dataset**.

```text
Input Image
     ↓
Image Preprocessing
     ↓
Pretrained ResNet50
     ↓
Global Max Pooling
     ↓
Feature Embedding
     ↓
L2 Normalization
```

---

## 2. Feature Database Creation

The current feature database contains:

```text
44,441 catalog images
```

Each image is processed through the same ResNet50 pipeline.

The generated embeddings and corresponding image paths are stored in:

```text
image_features_embedding.pkl
img_files.pkl
```

The large embedding file is managed using **Git LFS**.

```text
44,441 Catalog Images
        ↓
Image Preprocessing
        ↓
Pretrained ResNet50
        ↓
Global Max Pooling
        ↓
L2 Normalization
        ↓
Feature Embeddings
        ↓
image_features_embedding.pkl
```

---

## 3. Similarity Search

The query embedding is compared with the catalog embeddings using:

```text
Library  : scikit-learn
Algorithm: NearestNeighbors
Search   : brute-force
Metric   : Euclidean distance
```

Because the embeddings are L2-normalized, Euclidean distance and cosine similarity produce the same ranking for the current feature representation.

A **smaller Euclidean distance indicates a closer embedding match**.

---

## 4. Top-K Retrieval

The application supports:

```text
Top 5
Top 10
Top 15
```

The retrieved products are displayed dynamically in the Streamlit interface.

Each result also displays its Euclidean distance.

---

# 🧠 System Architecture

## Catalog Feature Generation

```text
Fashion Product Catalog
        ↓
Image Preprocessing
        ↓
Pretrained ResNet50
        ↓
Global Max Pooling
        ↓
L2 Normalization
        ↓
Feature Database
       / \
      /   \
Embeddings  Image Paths
```

## Query and Retrieval

```text
User Query Image
       ↓
Image Preprocessing
       ↓
Pretrained ResNet50
       ↓
Query Embedding
       ↓
L2 Normalization
       ↓
NearestNeighbors
       ↓
Euclidean Distance
       ↓
Top-K Similar Products
```

---

# 📂 Dataset

The project uses the:

**Fashion Product Images Dataset**

Dataset source:

https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset

The current feature database contains:

```text
44,441 product images
```

The full dataset is **not stored in this GitHub repository**.

Expected local structure:

```text
fashion_small/
└── images/
```

Example:

```text
fashion_small/images/1554.jpg
fashion_small/images/10000.jpg
fashion_small/images/11263.jpg
```

### Dataset Metadata

The dataset metadata contains fields such as:

- Product ID
- Gender
- Master Category
- Sub Category
- Article Type
- Base Colour
- Season
- Usage
- Product Display Name

For the current retrieval evaluation, **`articleType` is used as the relevance criterion**.

> `articleType` is a category-based proxy for relevance. It is not a perfect ground-truth measure of human-perceived visual similarity.

---

# 📊 Evaluation

The baseline retrieval system was evaluated using:

```text
Catalog Size        : 44,441 images
Evaluation Queries  : 500 images
Relevance Criterion : articleType
```

The same deterministic sample of 500 queries was used for the baseline experiments.

## Baseline Results

| Metric | @5 | @10 | @15 |
|---|---:|---:|---:|
| **Precision** | **77.96%** | **74.76%** | **73.79%** |

### Interpretation

The baseline achieved:

- **77.96% Precision@5**
- **74.76% Precision@10**
- **73.79% Precision@15**

These values measure how often the retrieved products belong to the same `articleType` category as the query.

Because `articleType` is only a proxy relevance label, these metrics should be interpreted as **category-consistency retrieval performance**, not human-judged visual similarity.

---

# 🔬 Distance Metric Experiment

Two distance metrics were evaluated using the same:

- 44,441 embeddings
- 500 evaluation queries
- `articleType` relevance criterion
- L2-normalized embeddings

| Distance Metric | Precision@5 | Precision@10 | Precision@15 |
|---|---:|---:|---:|
| **Euclidean** | **77.96%** | **74.76%** | **73.79%** |
| **Cosine** | **77.96%** | **74.76%** | **73.79%** |

### Conclusion

Both metrics produced the same retrieval ranking for the current L2-normalized embeddings.

Therefore, the production application continues to use:

```text
NearestNeighbors
+
Euclidean distance
```

---

# 🔬 Pooling Experiment

A second experiment compared:

```text
Global Max Pooling
vs.
Global Average Pooling
```

using the same 44,441-image catalog and the same evaluation methodology.

| Feature Representation | Precision@5 | Precision@10 | Precision@15 |
|---|---:|---:|---:|
| **Global Max Pooling** | **77.96%** | **74.76%** | **73.79%** |
| Global Average Pooling | 76.36% | 73.58% | 71.99% |

### Conclusion

Global Max Pooling performed better across all three Precision@K measurements.

Therefore, the production system retains:

```text
ResNet50
+
GlobalMaxPooling2D
```

---

# ✨ Features

- 📷 Upload a fashion/product image
- 🧠 Pretrained ResNet50 feature extraction
- 🔍 Content-Based Image Retrieval
- 🗂️ Search across 44,441 catalog images
- 📊 Top 5 / Top 10 / Top 15 recommendations
- 📏 Euclidean retrieval distance
- ⚡ Cached ResNet50 model
- 🔎 Nearest-neighbor retrieval
- 💾 Precomputed feature database
- 📈 Precision@K evaluation
- 🔬 Euclidean vs. cosine experiment
- 🔬 Global Max vs. Global Average Pooling experiment
- 🌐 Streamlit web interface
- 📦 Git LFS support
- 🧠 In-memory query image processing

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 **Python** | Application and data processing |
| 🧠 **TensorFlow / Keras** | ResNet50 feature extraction |
| 🔍 **scikit-learn** | Nearest-neighbor retrieval |
| 🔢 **NumPy** | Numerical computation |
| 🐼 **Pandas** | Metadata and evaluation |
| 🖼️ **Pillow** | Image processing |
| 🌐 **Streamlit** | Web application |
| 📦 **Git LFS** | Large embedding management |

---

# 📁 Project Structure

```text
Fashion-Recommendation-System-H/
│
├── app.py
├── main.py
├── evaluation_metadata.py
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
│
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── evaluate_cosine.py
│   └── evaluate_avg_pooling_metrics.py
│
├── image_features_embedding.pkl
├── img_files.pkl
│
├── Demo/
├── sample/
└── uploader/
```

### Main Files

| File | Purpose |
|---|---|
| `app.py` | Generates ResNet50 embeddings |
| `main.py` | Streamlit recommendation application |
| `image_features_embedding.pkl` | Precomputed production feature embeddings |
| `img_files.pkl` | Corresponding catalog image paths |
| `evaluation/evaluate_retrieval.py` | Baseline retrieval evaluation |
| `evaluation/evaluate_cosine.py` | Cosine-distance experiment |
| `evaluation/evaluate_avg_pooling_metrics.py` | Average-pooling experiment |
| `evaluation_metadata.py` | Metadata parsing |
| `requirements.txt` | Python dependencies |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/hssquare/Fashion-Recommendation-System-H.git

cd Fashion-Recommendation-System-H
```

## 2. Install Git LFS

Git LFS is required for the large embedding file.

```bash
git lfs install

git lfs pull
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

## 4. Activate the Environment

### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset Setup

Create the expected directory:

```text
fashion_small/
└── images/
```

Place the catalog images inside:

```text
fashion_small/images/
```

Verify the directory:

```powershell
Test-Path "fashion_small\images"
```

Expected output:

```text
True
```

Verify the catalog size:

```powershell
(Get-ChildItem "fashion_small\images" -File).Count
```

Expected output:

```text
44441
```

---

# 🧮 Generate Feature Embeddings

To regenerate the production feature database:

```bash
python app.py
```

The script:

1. Loads the ImageNet-pretrained ResNet50 feature extractor.
2. Reads images from `fashion_small/images`.
3. Extracts visual feature embeddings.
4. Applies Global Max Pooling.
5. Applies L2 normalization.
6. Saves `image_features_embedding.pkl`.
7. Saves `img_files.pkl`.

> Processing the complete 44,441-image catalog can take significant time on a CPU.

---

# ▶️ Run the Application

Start Streamlit:

```bash
python -m streamlit run main.py
```

Open:

```text
http://localhost:8501
```

Upload an image and select:

```text
5
10
15
```

recommendations.

---

# 🌐 Live Demo

Try the deployed application:

https://fashion-recommendation-system-h8w48qvsj6zbbnwyrtmk.streamlit.app/

---

# 🧪 Run Evaluation

### Baseline Evaluation

```bash
python evaluation/evaluate_retrieval.py
```

The baseline evaluation uses a deterministic sample of 500 queries and reports:

```text
Precision@5
Precision@10
Precision@15
```

### Cosine Distance Experiment

```bash
python evaluation/evaluate_cosine.py
```

This compares cosine distance with the Euclidean baseline.

### Average Pooling Experiment

The Average Pooling experiment requires the generated experimental embeddings and is intended for model comparison rather than production inference.

---

# ⚠️ Limitations

- 🖼️ The current production feature database uses a low-resolution image subset.
- 📐 Current catalog images are approximately **60×80 pixels**.
- 🧠 ResNet50 is an ImageNet-pretrained frozen feature extractor.
- 🏋️ No fashion-specific fine-tuning is currently performed.
- 🏷️ `articleType` is only a proxy relevance label.
- 👁️ Category similarity does not always equal visual similarity.
- 📊 Retrieval quality can vary between product categories and image styles.
- 💾 The complete dataset is not stored in GitHub.
- 🚫 The system does not use price, brand, descriptions, ratings, purchase history, or user preferences.

---

# 🔮 Future Improvements

- 🎯 Fine-tune a CNN using fashion-specific data
- 🖼️ Evaluate higher-resolution product images
- 🧠 Experiment with stronger image-embedding models
- 👗 Use fashion-specific pretrained models
- 🏷️ Add category-aware retrieval
- 🔎 Add metadata filtering
- 👥 Add human-annotated relevance judgments
- ⚡ Improve batch feature extraction
- 🧪 Add automated tests
- 📊 Add experiment tracking
- 📦 Add model/version management
- 🌐 Continue improving the public web demo

---

# 🔖 Release

The current stable release is:

```text
v1.0.0
```

Release name:

> **Fashion Vision v1.0.0**

Release:

https://github.com/hssquare/Fashion-Recommendation-System-H/releases/tag/v1.0.0

---

# 📊 Project Snapshot

```text
🗂️ Catalog
44,441 images

🧪 Evaluation Queries
500

🎯 Precision@5
77.96%

🎯 Precision@10
74.76%

🎯 Precision@15
73.79%

🧠 Backbone
ResNet50

🏊 Pooling
Global Max Pooling

📐 Normalization
L2

🔎 Search
NearestNeighbors

📏 Distance
Euclidean

🌐 Interface
Streamlit
```

---

# 🧠 Production Pipeline

```text
📷 Query Image
      ↓
🧹 Image Preprocessing
      ↓
🧠 ResNet50
      ↓
🏊 Global Max Pooling
      ↓
📐 L2 Normalization
      ↓
🔢 Visual Embedding
      ↓
🔎 NearestNeighbors
      ↓
📏 Euclidean Distance
      ↓
🏆 Top-K Products
```

---

# ✅ Conclusion

This project demonstrates a complete **Content-Based Image Retrieval** pipeline:

```text
Deep Visual Feature Extraction
            ↓
Embedding Normalization
            ↓
Nearest-Neighbor Search
            ↓
Top-K Retrieval
            ↓
Quantitative Evaluation
```

The current production system uses:

```text
Pretrained ResNet50
        ↓
Global Max Pooling
        ↓
L2 Normalization
        ↓
NearestNeighbors
        ↓
Euclidean Distance
```

The baseline achieved:

```text
Precision@5  = 77.96%

Precision@10 = 74.76%

Precision@15 = 73.79%
```

on 500 evaluation queries using `articleType` as the category-based relevance criterion.

---

# 👤 Author

## Harshal

GitHub:

https://github.com/hssquare

Repository:

https://github.com/hssquare/Fashion-Recommendation-System-H

Live Demo:

https://fashion-recommendation-system-h-ehw48qv8sjx6zbbnwyrtmk.streamlit.app/

---

## 🤗 Hugging Face Catalog

https://huggingface.co/datasets/GangHitman/fashion-recommendation-images

---

**Made by Harshal — Fashion Vision v1.0.0**