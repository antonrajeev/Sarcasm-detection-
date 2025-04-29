##Evaluating the Performance of Embedding Techniques for Sarcasm Detection in News Headlines



This repository contains the code and resources for the MSc Data Science project **"Evaluating the Performance of Embedding Techniques for Sarcasm Detection in News Headlines"**. The aim of the project is to analyze and compare embedding techniques — FastText, BERT, and RoBERTa — for the task of sarcasm detection in short text data (news headlines), using both machine learning and deep learning models.

## 🔍 Project Overview

Sarcasm detection is a critical task in NLP applications such as sentiment analysis, fake news detection, and chatbot systems. Sarcasm often relies on contextual or subtle cues that are hard to capture using traditional methods. This study evaluates how well different embeddings capture sarcasm and support accurate classification.

### 🎯 Objectives

- Preprocess and clean the dataset
- Apply FastText, BERT, and RoBERTa embeddings
- Train ML models (SVM, Random Forest, Logistic Regression)
- Train DL models (LSTM, BiLSTM)
- Evaluate using Accuracy, Precision, Recall, F1-score, and Macro F1-score

---

## 📁 Dataset

- **Name:** [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- **Format:** JSON
- **Size:** 26,709 headlines
- **Classes:** Sarcastic (`1`) and Non-sarcastic (`0`)

---

## ⚙️ Tech Stack

- Python (v3.8+)
- Transformers (HuggingFace)
- Scikit-learn
- TensorFlow / Keras
- NLTK
- Matplotlib / Seaborn

---

## 🧪 Models Evaluated

### Embedding Techniques

- [x] FastText
- [x] BERT
- [x] RoBERTa

### Classifiers

| Model             | BERT Acc. | RoBERTa Acc. | FastText Acc. |
|------------------|-----------|--------------|---------------|
| SVM              | 0.72      | 0.70         | <0.60         |
| Random Forest    | 0.71      | 0.69         | <0.60         |
| Logistic Reg.    | 0.68      | 0.66         | <0.60         |
| BiLSTM (DL)      | 0.72      | 0.70         | 0.59          |
| LSTM (DL)        | 0.71      | 0.68         | 0.57          |

---

## 🧰 How to Run

### 1. Clone this repo
```bash
git clone https://github.com/yourusername/sarcasm-detection.git
cd sarcasm-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) and place it in the root folder as `Sarcasm_Headlines_Dataset.json`.

### 4. Run preprocessing
```python
python preprocess.py
```

### 5. Train models
```python
python train_ml.py    # For SVM, RF, LR
python train_dl.py    # For LSTM, BiLSTM
```

---

## 📊 Results

- **Best ML Model:** SVM with BERT (Accuracy: 0.72, Macro F1: 0.72)
- **Best DL Model:** BiLSTM with BERT (Accuracy: 0.72, Macro F1: 0.72)
- **Worst Performer:** FastText across all models

---

## 📌 Limitations

- PCA reduced dimensions from 768/300 to 10, possibly losing semantic information.
- FastText lacks deep contextual understanding.
- Transformer models like BERT and RoBERTa are resource-intensive.

---

## 🚀 Future Work

- Use DistilBERT or GPT/T5 for comparison
- Integrate multimodal sarcasm detection (text + image)
- Explore hybrid embedding strategies

