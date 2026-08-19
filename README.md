# 🖼️ Image Similarity Finder 

With the increasing variety of fashion products available online, customers often struggle to find items visually similar to what they have in mind. This project addresses that challenge by building an **Image Similarity Finder** that retrieves visually similar fashion items based on a user-provided image. Instead of relying on user purchase history or ratings, this system uses **Content-Based Image Retrieval (CBIR)** techniques powered by deep learning to match products based on their visual features.

---

## 📌 Introduction
The solution allows users to upload an image of any fashion product and instantly find visually similar items from a product catalog. This is achieved by extracting feature embeddings of images using **Convolutional Neural Networks (CNNs)** and performing a nearest-neighbor search. Such systems help improve product discovery in e-commerce by enabling intuitive image-based search.

---

## ⚙️ Proposed Methodology
1. **Feature Extraction:** Pre-trained CNN models (**Pretrained ResNet50 used as a frozen feature extractor learninG H**) are used to extract high-dimensional feature embeddings from product images.  
2. **Database Creation:** Each item in the catalog is processed to store its feature vector in the database.  
3. **Similarity Search:** When a user uploads an image, its feature vector is computed and compared against the stored feature embeddings using Euclidean distance with scikit-learn's Nearest Neighbors.
4. **Top Matches:** The system returns the **top 5 most visually similar images**.  

---

## 📂 Dataset
The project uses the **[Fashion Product Images Dataset (Kaggle)](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)** for training and evaluation, which contains thousands of product images across multiple categories.

---

## 📊 Results
- Achieved **high retrieval accuracy** with visually coherent matches.
- Generated visual embeddings for 44,441 fashion product images.
- Used a pretrained ResNet50 model as a frozen feature extractor.
- Used scikit-learn NearestNeighbors for similarity retrieval.
- Returns the top 5 visually similar products for an uploaded query image.  
- Leveraged **transfer learning** and `scikit-learn`'s **Nearest Neighbors** for efficient search.  
- Can retrieve and display results with **low latency**.  

---

## 🛠 Tech Stack
- **TensorFlow / Keras** – Deep learning and transfer learning  
- **scikit-learn** – Nearest Neighbor search  
- **OpenCV, Pillow** – Image processing  
- **Streamlit** – Web-based front-end for image uploads and results display  

---

## ✅ Conclusion
The **Image Similarity Finder** offers a simple yet powerful way to search fashion products by images, improving user experience in e-commerce platforms. By using **CNN-based feature extraction** and **content-based image retrieval**, it provides accurate matches without relying on user behavior data.

---

