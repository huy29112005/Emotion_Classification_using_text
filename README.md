# Emotion Classification using BERT with Custom Backbone

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)

## 📌 Project Overview
This project focuses on identifying human emotions from text using a hybrid deep learning architecture. We utilize **BERT (Bidirectional Encoder Representations from Transformers)** as a pre-trained backbone for feature extraction and integrate it with a **custom-designed neural network head** to improve classification performance for specific emotion categories.



## 🏗️ Model Architecture
Unlike standard fine-tuning, this project implements a custom architecture on top of the transformer outputs to better capture emotional nuances:

* **Pre-trained Backbone:** `bert-base-uncased` serves as the primary feature extractor, providing rich contextual embeddings.
* **Custom Backbone/Head:**
    * **Hidden State Selection:** Optimized by utilizing the `[CLS]` token output.
    * **Custom Layers:** Integrated additional Dense layers and Dropout mechanisms to refine the feature space.
    * **Activation:** ReLU/GELU activation functions with a final **Softmax** layer for multi-class classification.

## 📊 Dataset
* **Source:** [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
* **Categories:** * 0: **Sadness**
    * 1: **Joy**
    * 2: **Love**
    * 3: **Anger**
    * 4: **Fear**
    * 5: **Surprise**
* **Pre-processing:** Tokenization, Padding, and Attention Masking performed via Hugging Face `BertTokenizer`.

## 🚀 Key Features
* **Hybrid Approach:** Combines state-of-the-art Transformers with custom architectural tweaks.
* **Transfer Learning:** Efficiently adapts a massive pre-trained model to a specific niche task.
* **High Performance:** Optimized to handle informal language, slang, and short-text contexts typical of emotional expression.

## 🛠️ Installation & Usage

### 1. Clone the Repo
```bash
git clone [https://github.com/your-username/emotion-bert-custom.git](https://github.com/your-username/emotion-bert-custom.git)
cd emotion-bert-custom
