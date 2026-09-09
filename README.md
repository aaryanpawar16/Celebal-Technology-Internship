# Celebal Technologies Internship — Data Science & ML

This repository contains my weekly assignments and capstone project for the Celebal Technologies internship, covering the full spectrum of the data science and machine learning pipeline: classical ML, deep learning (CNNs, RNN/LSTM/GRU, autoencoders), and modern LLM-based systems (RAG, knowledge graphs, and agentic pipelines with LangGraph).

## 📁 Repository Structure

```
├── week 1/    → ML Foundations
├── week 2/    → Tesla Deliveries & Production ML Pipeline
├── week 3/    → Customer Intelligence System (Clustering + Classification)
├── week 4/    → CIFAR-10 Image Classification (ANN vs CNN)
├── week 5/    → Deep Learning Text Generation (RNN vs LSTM vs GRU)
├── week 6/    → MNIST Image Denoising (Convolutional Autoencoder)
├── week 7/    → RAG Pipeline — Document Question Answering System
├── week 8/    → Single Agent Pipeline Project
└── Memory-Augmented Chatbot/   → Capstone Project (RAG + Knowledge Graph + LangGraph)
```

---

## Week 1 — ML Foundations

Introductory assignment covering the core building blocks of the machine learning workflow: data loading, inspection, cleaning, and foundational modeling concepts that set up the pipelines built in later weeks.

📂 [`week 1/`](./week%201)

---

## Week 2 — Tesla Deliveries & Production ML Pipeline

An end-to-end regression pipeline on Tesla delivery and production data (2015–2025): EDA, feature engineering (lag and rolling-mean features), a chronological train/test split, Linear Regression and GridSearchCV-tuned Random Forest models, 5-fold cross-validation, an ADF stationarity test, and a full forecast comparison.

**Highlights:** Linear Regression R² = 0.9908, Random Forest (tuned) R² = 0.9902, 5-fold CV mean R² = 0.9903.

📂 [`week 2/`](./week%202)

---

## Week 3 — Customer Intelligence System

An unsupervised-to-supervised pipeline on country-level socioeconomic data: K-Means and DBSCAN clustering (with PCA visualization) to segment countries into "developed / developing / least developed" groups, followed by Random Forest, XGBoost, and a soft-voting ensemble trained to classify countries into their K-Means segment.

**Highlights:** K-Means silhouette ≈ 0.283; Random Forest 100% test accuracy, Voting Ensemble 97% test accuracy predicting cluster membership.

📂 [`week 3/`](./week%203)

---

## Week 4 — CIFAR-10 Image Classification (ANN vs CNN)

A comparison of an Artificial Neural Network and a Convolutional Neural Network on CIFAR-10, demonstrating why CNNs outperform ANNs on image data, plus experiments with dropout, batch normalization, and data augmentation.

**Highlights:** ANN test accuracy 44.4%, CNN test accuracy 68.6%.

📂 [`week 4/`](./week%204)

---

## Week 5 — Deep Learning Text Generation

A comparison of Vanilla RNN, LSTM, and GRU architectures for next-word prediction and text generation, including an extended experiment with a larger corpus, wider hidden units, and more training epochs to see how architectural differences narrow with scale.

📂 [`week 5/`](./week%205)

---

## Week 6 — MNIST Image Denoising

A convolutional autoencoder trained to remove Gaussian noise from MNIST digit images, evaluated both visually and quantitatively via PSNR.

**Highlights:** PSNR improved from 10.98 dB (noisy) to 20.85 dB (denoised).

📂 [`week 6/`](./week%206)

---

## Week 7 — RAG Pipeline: Document Question Answering System

A Retrieval-Augmented Generation pipeline that answers questions over a document corpus by retrieving relevant chunks and grounding an LLM's response in that retrieved context.

📂 [`week 7/`](./week%207)

---

## Week 8 — Single Agent Pipeline Project

An agentic pipeline built around a single LLM-driven agent capable of reasoning through a task and invoking tools to complete it.

📂 [`week 8/`](./week%208)

---

## 🏆 Capstone Project — Memory-Augmented Chatbot (RAG + Knowledge Graph + LangGraph)

The internship's capstone project brings together the skills developed across all eight weeks into a single system: a chatbot with persistent memory that combines:

- **RAG (Retrieval-Augmented Generation)** — grounding responses in relevant retrieved documents/context
- **Knowledge Graph** — structured, relational memory that captures entities and their relationships over time, going beyond flat vector retrieval
- **LangGraph** — an agentic orchestration layer that manages the chatbot's multi-step reasoning, tool use, and state across a conversation

Together, these components let the chatbot maintain and reason over long-term conversational memory rather than treating each exchange in isolation.

📂 [`Memory-Augmented Chatbot/`](./Memory-Augmented%20Chatbot)

---

## Tech Stack (across the repo)

- **Classical ML:** scikit-learn, XGBoost
- **Deep Learning:** TensorFlow / Keras
- **Data handling & visualization:** pandas, numpy, matplotlib, seaborn
- **LLM / Agentic tooling:** RAG pipelines, knowledge graphs, LangGraph

## Author

**Aaryan Pawar**
Celebal Technologies Internship — Data Science
