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
[![Release](https://img.shields.io/badge/Release-v1.0.0-00C853)](https://github.com/hssquare/Fashion-Recommendation-System-H/releases/tag/v1.0.0)

> 🧠 A deep-learning-based **Content-Based Image Retrieval (CBIR)** system that finds visually similar fashion products from a catalog of **44,441 images**.

## 🚀 Live Demo

**Fashion Vision — Streamlit App**

https://fashion-recommendation-system-h8w48qvsj6zbbnwyrtmk.streamlit.app/

## 🐙 GitHub Repository

https://github.com/hssquare/Fashion-Recommendation-System-H

## 🤗 Hugging Face Catalog

https://huggingface.co/datasets/GangHitman/fashion-recommendation-images

## 🏷️ Current Release

**Fashion Vision v1.0.0**

---

# 📌 Table of Contents

- [📌 Overview](#-overview)
- [🎯 Problem Statement](#-problem-statement)
- [🎯 Project Objectives](#-project-objectives)
- [📊 Project Highlights](#-project-highlights)
- [✨ Key Features](#-key-features)
- [🧠 What Type of System Is This?](#-what-type-of-system-is-this)
- [⚙️ Methodology](#-methodology)
- [🖼️ Image Preprocessing](#-1--image-preprocessing)
- [🧠 Feature Extraction](#-2--feature-extraction)
- [❄️ Frozen ResNet50](#-3--frozen-resnet50)
- [🏊 Global Max Pooling](#-4--global-max-pooling)
- [📐 L2 Normalization](#-5--l2-normalization)
- [💾 Feature Database](#-6--feature-database)
- [🔎 Similarity Search](#-7--similarity-search)
- [🔢 Top-K Retrieval](#-8--top-k-retrieval)
- [🖼️ High-Resolution Product Catalog](#-9--high-resolution-product-catalog)
- [🖥️ Cross-Platform Path Handling](#-10--cross-platform-path-handling)
- [🏗️ System Architecture](#-system-architecture)
- [📂 Dataset](#-dataset)
- [📁 Dataset Structure](#-dataset-structure)
- [🏷️ Dataset Metadata](#-dataset-metadata)
- [📊 Evaluation](#-evaluation)
- [🎯 Precision@K](#-precisionk)
- [🏆 Baseline Results](#-baseline-results)
- [🔬 Experiment 1 — Euclidean vs Cosine](#-experiment-1--euclidean-vs-cosine)
- [🔬 Experiment 2 — Global Max vs Global Average Pooling](#-experiment-2--global-max-vs-global-average-pooling)
- [🛠️ Production Configuration](#-production-configuration)
- [⚡ Performance Improvements](#-performance-improvements)
- [🛠️ Technology Stack](#-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Installation](#-installation)
- [📂 Dataset Setup](#-dataset-setup)
- [🧮 Generate Embeddings](#-generate-embeddings)
- [▶️ Run Locally](#-run-locally)
- [🧪 Run Evaluation](#-run-evaluation)
- [📦 Git LFS](#-git-lfs)
- [🌐 Deployment](#-deployment)
- [🔖 Release](#-release)
- [⚠️ Limitations](#-limitations)
- [🔮 Future Improvements](#-future-improvements)
- [🛠️ Troubleshooting](#-troubleshooting)
- [💼 Resume Description](#-resume-description)
- [📚 Learning Outcomes](#-learning-outcomes)
- [🧪 Reproducibility](#-reproducibility)
- [🏁 Conclusion](#-conclusion)
- [👤 Author](#-author)

---

# 📌 Overview

The **Fashion Recommendation System** is an image-based fashion product retrieval application.

Instead of asking the user to type a product name, category, or keyword, the system allows the user to upload an image.

The uploaded image is converted into a deep visual embedding using an ImageNet-pretrained **ResNet50** model.

That embedding is compared against a precomputed catalog embedding database using nearest-neighbor search.

The closest products are then retrieved and displayed.

The deployed version additionally resolves retrieved product IDs to high-resolution product images from a Hugging Face catalog.

---

# 🎯 Problem Statement

Traditional recommendation systems often depend on:

- User ratings
- Purchase history
- Click history
- User preferences
- Collaborative filtering
- Product metadata

Those systems are useful when interaction data is available.

However, a visual product search problem is different.

A user may see a product in an image and want to discover products that look similar.

For example:

```text
📷 User sees a blue shirt
        ↓
📤 Uploads the image
        ↓
🧠 System extracts visual features
        ↓
🔎 Searches the fashion catalog
        ↓
🏆 Returns visually similar products