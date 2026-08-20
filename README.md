# 🖼️ Image Similarity Finder

An image-based fashion product retrieval system built using **Content-Based Image Retrieval (CBIR)** and deep visual feature extraction.

Users can upload a fashion or product image and retrieve visually similar products from a catalog using a pretrained **ResNet50** model and nearest-neighbor search.

---

## 📌 Introduction

Traditional recommendation systems often depend on ratings, purchase history, or collaborative filtering.

This project uses the **uploaded image itself as the query**.

The query image is converted into a visual feature embedding using a pretrained **ResNet50 CNN**. The embedding is L2-normalized and compared with precomputed catalog embeddings to retrieve visually similar products.

The application is implemented using **Streamlit**.

---

## 🎯 Project Objective

- 📷 Accept a fashion/product image
- 🧠 Extract visual features using ResNet50
- 🔍 Search a large product catalog
- 🏷️ Retrieve visually similar products
- 📊 Rank and display recommendations
- 🌐 Provide an interactive web interface

---

## ⚙️ Proposed Methodology

### 1️⃣ Feature Extraction

A pretrained **ResNet50** model with ImageNet weights is used as a **frozen feature extractor**.

The classification head is removed and **Global Max Pooling** is applied to generate a visual feature representation.

The current implementation does **not** train or fine-tune ResNet50 on the fashion dataset.

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

### 2️⃣ Feature Database Creation

The catalog contains **44,441 product images**.

Each image is processed through the same ResNet50 pipeline.

The resulting embeddings and image paths are stored in:

```text
image_features_embedding.pkl
img_files.pkl
```

The large embedding file is tracked using **Git LFS**.

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

### 3️⃣ Similarity Search

The query embedding is compared with catalog embeddings using
**scikit-learn NearestNeighbors**.

Current configuration:

```text
Algorithm : brute-force nearest-neighbor search
Metric    : Euclidean distance
```

Because the embeddings are L2-normalized, Euclidean distance and cosine similarity produce the same ranking for the current representation.

### 4️⃣ Top-K Retrieval

Users can select:

- 🥇 Top 5
- 🥈 Top 10
- 🥉 Top 15

A smaller Euclidean distance indicates a closer embedding match.

---

## 🧠 System Architecture

### 📦 Catalog Feature Generation

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
   ┌────┴────┐
   ↓         ↓
Embeddings  Image Paths
```

### 🔎 Query and Retrieval

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
Top-K Results
```

---

## 📂 Dataset

The project uses the **[Fashion Product Images Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)** from Kaggle.

Current feature database:

```text
44,441 product images
```

The original dataset is **not stored in this GitHub repository** because of its size.

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

### 🏷️ Dataset Metadata

The metadata includes:

- Product ID
- Gender
- Master Category
- Sub Category
- Article Type
- Base Colour
- Season
- Usage
- Product Display Name

`articleType` is used as the relevance criterion during evaluation.

---

## 📊 Evaluation

Evaluation configuration:

```text
Catalog Size        : 44,441 images
Evaluation Queries  : 500 images
Relevance Criterion : articleType
```

`articleType` is a **category-based proxy for relevance**, not a perfect ground-truth measure of human-perceived visual similarity.

### 📈 Retrieval Metrics

| Metric | @5 | @10 | @15 |
|---|---:|---:|---:|
| **Precision** | **77.96%** | **74.76%** | **73.79%** |
| **Recall** | **0.86%** | **1.46%** | **2.01%** |
| **mAP** | **73.20%** | **68.57%** | **66.61%** |

### 📌 Interpretation

The baseline achieved:

- ✅ **77.96% Precision@5**
- ✅ **74.76% Precision@10**
- ✅ **73.79% Precision@15**
- ✅ **73.20% mAP@5**

These results represent category-consistent retrieval under the selected `articleType` relevance definition.

---

## 🔬 Distance Metric Experiment

Two metrics were evaluated using the same:

- 44,441 embeddings
- 500 queries
- `articleType` relevance criterion
- L2-normalized embeddings

| Distance Metric | Precision@5 | Precision@10 | Precision@15 |
|---|---:|---:|---:|
| **Euclidean** | **77.96%** | **74.76%** | **73.79%** |
| **Cosine** | **77.96%** | **74.76%** | **73.79%** |

### ✅ Conclusion

Euclidean distance and cosine similarity produced the same ranking for the current normalized embeddings.

The application therefore continues to use **Euclidean distance with NearestNeighbors**.

---

## ✨ Features

- 📷 Upload a fashion/product image
- 🧠 Pretrained ResNet50 feature extraction
- 🔍 Content-Based Image Retrieval
- 🗂️ Search across 44,441 catalog images
- 📊 Top 5 / Top 10 / Top 15 recommendations
- 📏 Display Euclidean distance
- ⚡ Cached ResNet50 model
- 🔎 Nearest-neighbor retrieval
- 💾 Precomputed feature database
- 📈 Precision, Recall, and mAP evaluation
- 🔬 Euclidean vs. cosine comparison
- 🌐 Streamlit interface
- 💾 Git LFS support
- 🧠 In-memory query image processing

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 **Python** | Application and data processing |
| 🧠 **TensorFlow / Keras** | ResNet50 feature extraction |
| 🔍 **scikit-learn** | Nearest-neighbor retrieval |
| 🔢 **NumPy** | Numerical computation |
| 🐼 **Pandas** | Metadata and evaluation |
| 🖼️ **Pillow** | Image processing |
| 👁️ **OpenCV** | Image processing |
| 🌐 **Streamlit** | Web application |
| 📦 **Git LFS** | Large embedding management |

---

## 📁 Project Structure

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
│   └── evaluate_cosine.py
│
├── image_features_embedding.pkl
├── img_files.pkl
│
├── Demo/
├── sample/
└── uploader/
```

### 📄 Main Files

| File | Purpose |
|---|---|
| `app.py` | Generates ResNet50 embeddings |
| `main.py` | Streamlit recommendation application |
| `image_features_embedding.pkl` | Precomputed feature embeddings |
| `img_files.pkl` | Corresponding image paths |
| `evaluation/evaluate_retrieval.py` | Precision, Recall and mAP |
| `evaluation/evaluate_cosine.py` | Cosine-distance experiment |
| `evaluation_metadata.py` | Metadata parsing |
| `requirements.txt` | Python dependencies |

---

## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/hssquare/Fashion-Recommendation-System-H.git
cd Fashion-Recommendation-System-H
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate the Environment

#### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Git LFS Setup

The repository uses **Git LFS** for the large embedding file.

```bash
git lfs install
git lfs pull
```

This retrieves the actual LFS-managed embedding file:

```text
image_features_embedding.pkl
```

---

## 📂 Dataset Setup

The complete dataset is not stored in GitHub.

Create:

```text
Fashion-Recommendation-System-H/
└── fashion_small/
    └── images/
```

Place the catalog images inside `fashion_small/images/`.

---

## 🧮 Generate Feature Embeddings

To regenerate the feature database:

```bash
python app.py
```

The script:

1. Loads the pretrained ResNet50 feature extractor.
2. Reads images from `fashion_small/images`.
3. Extracts visual embeddings.
4. L2-normalizes the embeddings.
5. Saves `image_features_embedding.pkl`.
6. Saves `img_files.pkl`.

> Processing all 44,441 images can take significant time on a CPU.

---

## ▶️ Run the Application

Start Streamlit:

```bash
python -m streamlit run main.py
```

Open:

```text
http://localhost:8501
```

Upload an image and select 5, 10, or 15 recommendations.

---

## 🧪 Run Evaluation

Run:

```bash
python evaluation/evaluate_retrieval.py
```

Reports:

- Precision@5
- Precision@10
- Precision@15
- Recall@5
- Recall@10
- Recall@15
- mAP@5
- mAP@10
- mAP@15

The current evaluation uses a deterministic sample of 500 queries.

### 🔬 Cosine Experiment

```bash
python evaluation/evaluate_cosine.py
```

This compares cosine similarity with the Euclidean baseline.

---

## ⚠️ Limitations

- 🖼️ Current feature database uses a low-resolution image subset.
- 📐 Current catalog images are approximately **60×80 pixels**.
- 🧠 ResNet50 is an ImageNet-pretrained frozen feature extractor.
- 🏋️ No fashion-specific fine-tuning is currently performed.
- 🏷️ `articleType` is only a proxy relevance label.
- 👁️ Category similarity does not always equal visual similarity.
- 📊 Retrieval quality varies across categories and image styles.
- 💾 The complete dataset is not stored in GitHub.
- 🚫 The system does not currently use price, brand, descriptions, ratings, purchase history, or user preferences.

---

## 🔮 Future Improvements

- 🎯 Fine-tune a CNN on fashion-specific data
- 🖼️ Evaluate higher-resolution images
- 🧪 Compare Global Max Pooling with Global Average Pooling
- 🧠 Experiment with stronger image-embedding models
- 👗 Use fashion-specific pretrained models
- 🏷️ Add category-aware retrieval
- 🔎 Add metadata filtering
- 👥 Add human-annotated relevance judgments
- ⚡ Improve batch feature extraction
- 🧪 Add automated tests
- 📊 Add experiment tracking
- 📦 Add model/version management
- 🌐 Deploy a public web demo

---

## ✅ Conclusion

The **Image Similarity Finder** demonstrates a complete **Content-Based Image Retrieval** pipeline:

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

The project combines:

- 👁️ Computer Vision
- 🧠 Deep Learning
- 🔢 Feature Embeddings
- 🔎 Information Retrieval
- 🐍 Python
- 🧠 TensorFlow / Keras
- 🔍 scikit-learn
- 🌐 Streamlit

to provide image-based fashion product discovery without relying on ratings, purchase history, or collaborative filtering.

---














## 👤 Author

### **Harshal**

🔗 GitHub: https://github.com/hssquare

📦 Repository: https://github.com/hssquare/Fashion-Recommendation-System-H