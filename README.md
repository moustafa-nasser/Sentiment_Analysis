# 🛍️ Amazon Reviews Sentiment Analysis with LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## 📖 Overview
This project leverages **Deep Learning** and **Natural Language Processing (NLP)** to classify Amazon product reviews into **Positive** or **Negative** sentiments. It utilizes a **Long Short-Term Memory (LSTM)** neural network to capture sequential dependencies in text data, achieving remarkable accuracy. The project also includes a user-friendly web application built with **Streamlit** for real-time inference.

## 🚀 Features
* **Robust Preprocessing:** Includes text cleaning (regex), stopwords removal, and lemmatization using NLTK.
* **Deep Learning Model:** Implements a multi-layer LSTM architecture with Batch Normalization and Dropout to prevent overfitting.
* **High Performance:** Achieved an accuracy of approximately **89%** on the test dataset.
* **Interactive UI:** A deployed Streamlit application allowing users to input text and view sentiment predictions instantly.
* **Visualization:** Exploratory Data Analysis (EDA) including target distribution and text polarity analysis.

## 🛠️ Technologies Used
* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **NLP Libraries:** NLTK, TextBlob
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit

## 📂 Dataset
The model is trained on the **Amazon Reviews for Sentiment Analysis** dataset from Kaggle.
* **Source:** [Bittlingmayer/AmazonReviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
* **Size:** ~4 Million reviews (Sampled for training/testing in this project).
* **Labels:**
    * `__label__1` : Negative (Mapped to 0)
    * `__label__2` : Positive (Mapped to 1)

## 🧠 Model Architecture
The model is built using `Sequential` API from Keras:
1.  **Embedding Layer:** Converts words into dense vectors of fixed size.
2.  **LSTM Layer (128 units):** Captures long-term dependencies in text sequences (return_sequences=True).
3.  **BatchNormalization:** Stabilizes and accelerates training.
4.  **LSTM Layer (64 units):** Further processes the sequence features.
5.  **Dense Layer (64 units, ReLU):** Fully connected layer for feature interpretation.
6.  **Dropout (0.4):** Regularization to reduce overfitting.
7.  **Output Layer (Sigmoid):** Binary classification probability.

## 📊 Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | **89.32%** |
| **Precision (0 - Neg)** | 0.92 |
| **Recall (1 - Pos)** | 0.92 |
| **F1-Score** | 0.90 |

*(Metrics based on the test set evaluation)*

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/amazon-sentiment-analysis.git](https://github.com/your-username/amazon-sentiment-analysis.git)
cd amazon-sentiment-analysis
