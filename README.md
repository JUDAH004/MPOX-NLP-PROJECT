# 🦠 Natural Language Processing of Mpox-Related Instagram Captions: Sentiment Analysis & Hate Speech Detection with Gradio Deployment
## 🧾 Project Title
Understanding Public Sentiment and Online Toxicity Around the Mpox Outbreak Using NLP Techniques on Instagram Data

## 📜 Executive Summary
In this project, we leverage state-of-the-art Natural Language Processing (NLP) tools and machine learning algorithms to extract insights from a large-scale, multilingual Instagram dataset related to the global Mpox (formerly Monkeypox) outbreak. Our aim is twofold:

To classify the emotional sentiment embedded in user-generated Instagram posts.

To identify and flag hate speech, thereby helping public health stakeholders monitor and moderate digital discourse.

Using this real-world dataset, we trained robust classification models and deployed an interactive, user-friendly web application powered by Gradio. The app offers real-time sentiment and toxicity predictions and visual explanations of model decisions using LIME (Local Interpretable Model-Agnostic Explanations).

## 📍 Table of Contents
Background & Motivation

Business Understanding

Dataset Overview

Data Exploration & Key Findings

Data Preprocessing

Modeling Strategy

Evaluation & Metrics

Model Interpretability with LIME

Gradio-Based Web Deployment

User Guide for Web App

Insights & Recommendations

Limitations & Proposed Improvements

Project Architecture & File Structure

Credits

## 1. 🧭 Background & Motivation
The Mpox outbreak, which gained global attention in recent years, became a focal point for intense discourse on social media. As fear, misinformation, and stigma spread online, platforms such as Instagram transformed into arenas where both helpful and harmful content were shared at scale. Social listening using NLP offers public health authorities a means of systematically capturing the pulse of public opinion, emotional response, and toxicity—insights that are critical during global health emergencies.

## 2. 🎯 Business Understanding
The primary business problem is the lack of automated tools for monitoring public sentiment and online hate speech in real-time across social platforms. This project addresses this gap by:

Classifying the emotional tone of Instagram posts using a 7-class sentiment model.

Detecting hate speech to support content moderation and flag potentially harmful or stigmatizing language.

These tasks are not only technically challenging due to language ambiguity, sarcasm, and class imbalance but also socially significant given their implications for public well-being, crisis communication, and misinformation control.

## 3. 📊 Dataset Overview
Name: Mpox Instagram Dataset – Sentiment and Hate Analysis

Source: Kaggle Dataset by Sameep Vani

Records: 60,127 unique Instagram posts

Languages: Over 50 languages present; English comprises 97% of data

Attributes:

caption: Translated Instagram post

label: Sentiment category (neutral, fear, joy, sadness, anger, surprise, disgust)

hate: Binary indicator for hate speech (1 = hateful, 0 = non-hateful)

language: Original language of the post

## 4. 🔎 Data Exploration & Key Findings
Key insights derived from our exploratory data analysis include:

Class Imbalance: Hate speech represents only 4.25% of the dataset, requiring careful handling during modeling.

Text Length Distribution: Posts ranged widely in length, averaging ~547 characters.

Language Concentration: Though multilingual, 58,616 out of 60,127 posts were in English.

Emotion Distribution: Dominant emotions were fear, sadness, and anger, reflecting public anxiety and distrust.

We further limited our modeling scope to the top three languages—English, Finnish, and Spanish—to ensure consistency and quality during preprocessing and modeling.

## 5. 🧼 Data Preprocessing
The raw captions underwent several transformations:

Lowercasing and punctuation removal

Elimination of emojis and URLs

Tokenization and lemmatization via spaCy

Stopword removal

Feature engineering using TF-IDF vectorization

These steps ensured that the input data was clean, normalized, and meaningful for feature extraction and model training.

## 6. ⚙️ Modeling Strategy
We designed two separate pipelines:

A. Sentiment Classification
Model: XGBoost (Gradient Boosted Trees)

Classes: neutral, fear, joy, sadness, anger, surprise, disgust

Pipeline: TF-IDF Vectorizer → XGBoost Classifier

B. Hate Speech Detection
Model: LightGBM (Histogram-Based Gradient Boosting)

Classes: Hateful, Non-Hateful

Pipeline: TF-IDF Vectorizer → LightGBM Classifier

## 7. 📏 Evaluation & Metrics
Both models were evaluated using stratified train-test splits with metrics including:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

ROC-AUC (for binary classification)

Model performance was satisfactory across tasks, with the hate speech model achieving over 90% accuracy and sentiment classification delivering robust results across all emotional classes.

## 8. 🧠 Model Interpretability with LIME
To ensure our models are transparent and trustworthy, we integrated LIME to visualize how specific words influenced predictions.

For example:

A caption like “They are spreading lies about Mpox again!” highlighted "lies" and "again" as key drivers for both anger and hateful predictions.

This level of explanation is crucial in sensitive domains like public health, where misclassification can have real-world implications.

## 9. 🌍 Gradio-Based Web Deployment
We deployed our models using Gradio, which offers a sleek, interactive web interface for real-time NLP prediction.

Key Deployment Features:
Input: Text box for entering or pasting Instagram captions

Prediction Type: Drop-down menu to select sentiment or hate speech classification

Output:

Predicted class

Class probabilities

LIME visualization for interpretability

## 10. 🖱 User Guide for Web App
Open the Gradio application (locally or hosted).

Paste any Mpox-related Instagram caption into the input box.

Choose between:

Sentiment Analysis

Hate Speech Detection

View:

Predicted class

Probability breakdown

Word cloud visualization (via LIME) showing what influenced the decision

## 11. 📌 Insights & Recommendations
A significant portion of the public reaction to Mpox involved fear and sadness, indicating widespread uncertainty and vulnerability.

Hate speech, while not dominant, was still present and often used charged or stigmatizing language.

XGBoost and LightGBM, though relatively lightweight models, achieved excellent performance, making them suitable for deployment in low-resource environments.

## 12. 🚧 Limitations & Proposed Improvements
Current Limitations:
Only textual content was analyzed — no image or hashtag analysis

Restricted to top 3 languages, excluding valuable multilingual data

Used classical ML models — no transformer-based models like BERT or RoBERTa

Future Work:
Fine-tune transformer models (e.g., multilingual BERT)

Expand dataset to include comments and hashtags

Incorporate image captioning or computer vision

Host the app on Hugging Face Spaces or Streamlit Cloud