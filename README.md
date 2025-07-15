 # NLP on Mpox Instagram Posts: Sentiment & Hate Speech Detection + Gradio Deployment
## 📘 Overview
This project combines Natural Language Processing (NLP), machine learning, and Gradio-based deployment to analyze user-generated Instagram content about Mpox (formerly Monkeypox). The study seeks to understand public sentiment and detect hate speech, enabling health communicators and researchers to track emotional trends and online toxicity during public health crises.

## 🗂️ Table of Contents
1. Introduction

2. Business Understanding

3. Data Understanding

4. Data Preparation

5. Modeling Approach

6. Evaluation

7. Interpretability with LIME

8. Gradio Deployment

9. How to Use the App

10. Key Insights

11. Limitations & Future Work

12. File Structure

13. Credits

## 1. 🌍 Introduction
The Mpox outbreak demonstrated how misinformation, fear, and stigma spread on social media. Instagram, a highly visual and global platform, became a hotbed for both public expression and disinformation.

This project addresses the need for automated systems that can:

Monitor emotional sentiment

Detect toxic or hateful language

Assist public health stakeholders in creating safer online spaces

## 2. 🎯 Business Understanding
Project Objectives:

🧠 Detect emotional sentiments in Mpox-related Instagram posts:
neutral, fear, joy, sadness, anger, surprise, disgust

🚨 Detect hate speech in the same content:
Hateful or Non-Hateful

Why It Matters:

NGOs and health agencies can use these models to monitor online sentiment in real time.

Identifying hate speech helps reduce stigma and protects vulnerable populations.

Insights support policy development and crisis communication strategies.

## 3. 📊 Data Understanding
Source: Kaggle - Mpox Instagram Dataset
Records: ~60,127
Features:

caption: Instagram text post (translated if needed)

label: Sentiment category

hate: Binary hate speech label

language: Language of post

Key Observations:

97% of posts are in English, but over 50 languages represented

Only ~4.25% of posts are labeled as hateful → significant class imbalance

Average post length: ~547 characters

Focus for This Project:

Filtered to English, Finnish, and Spanish — top 3 languages

## 4. 🧹 Data Preparation
Text Preprocessing:

Lowercasing

URL and emoji removal

Tokenization using spaCy

Stopword removal

Lemmatization

Vectorization:

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert cleaned text into feature vectors.

Handling Class Imbalance:

Stratified splits for train/test

Evaluation with precision/recall to mitigate class skew

## 5. 🤖 Modeling Approach
Task 1: Sentiment Classification
Model: XGBoost

Inputs: TF-IDF vectors

Outputs: Multiclass prediction (7 classes)

Task 2: Hate Speech Detection
Model: LightGBM

Inputs: TF-IDF vectors

Outputs: Binary classification (Hateful / Not)

## 6. 📏 Evaluation
Task	Model	Accuracy	Precision	Recall	F1-Score
Sentiment Analysis	XGBoost	>85%	High	High	High
Hate Speech Detection	LightGBM	>90%	High	High	High

Confusion matrices, ROC curves, and classification reports are provided in the notebook.

## 7. 🔍 Interpretability with LIME
To promote transparency, we used LIME (Local Interpretable Model-Agnostic Explanations) to visualize how each word influences model predictions.

For example:

For a post like "They’re lying about the disease again!", LIME highlights "lying" and "disease" as key contributors to an angry or hateful label.

## 8. 🌐 Gradio Deployment
Why Gradio?

Simple Python interface for deploying ML models

Real-time interactivity with live predictions

Embeds easily in websites or internal dashboards

Features:

Upload or type text

Choose between:

Sentiment classification (via XGBoost)

Hate speech detection (via LightGBM)

Outputs:

Predicted label

Probabilities

LIME visualization of word impact

## 9. 🖱 How to Use the App
Enter Instagram caption text (related to Mpox)

Select prediction type:

"Sentiment Analysis"

"Hate Speech Detection"

View prediction output, including:

The predicted class

Probability distribution

LIME explanation plot

## 10. 📌 Key Insights
Fear and sadness dominate emotional responses to Mpox.

Hateful content, though sparse, contains emotionally charged language.

Instagram can serve as a barometer for public health anxiety.

Lightweight models like XGBoost and LightGBM are sufficient for high performance.

## 11. ⚠️ Limitations & Future Work
Current Limitations:
Only uses text (no image analysis)

Multilingual posts were reduced to just 3 languages

No deep learning models used (e.g., BERT)

Future Work:
Expand to multilingual support using mBERT or XLM-R

Add image captioning or OCR from post images

Integrate toxicity scoring using APIs like PerspectiveAPI

Deploy via Docker or HuggingFace Spaces


