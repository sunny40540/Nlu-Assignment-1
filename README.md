
# Sports vs Politics Text Classifier
**Natural Language Understanding – Assignment 1**  
**Name:** Sunny Kumar  
**Roll Number:**  B23CS1071

---

## Project Overview

This project implements a Natural Language Processing (NLP) text classification system that categorizes input text into one of two classes:

- **Sports**
- **Politics**

The objective of this assignment is to compare traditional machine learning algorithms and understand how well they perform on text classification tasks using simple feature engineering techniques.

Three classifiers are implemented and evaluated:

1. Naive Bayes
2. Support Vector Machine (SVM)
3. Logistic Regression

Instead of deep learning, this project focuses on classical machine learning approaches to demonstrate their effectiveness on small datasets.

---

## Dataset

The dataset is manually created and embedded inside the Python program.  
It contains short sentences describing either sports activities or political events.

### Sports Category
Includes sentences related to:
- matches and tournaments
- teams and players
- performance and results

### Politics Category
Includes sentences related to:
- government policies
- elections
- administration
- parliament discussions

### Dataset Size

| Category | Samples |
|--------|------|
| Sports | 20 |
| Politics | 20 |
| **Total** | **40** |

The dataset is balanced to avoid bias during training.

The dataset is split into:
- Training set: 30 samples
- Testing set: 10 samples

---

## Feature Extraction (TF-IDF)

Machine learning algorithms cannot understand raw text.  
Therefore, the text is converted into numerical form using:

**TF-IDF (Term Frequency – Inverse Document Frequency)**

TF-IDF assigns importance to words based on how unique they are in a category.

Examples:

| Word | Category Indicator |
|----|----|
goal | Sports
match | Sports
government | Politics
election | Politics

This allows the classifier to learn patterns from words.

---

## Machine Learning Models

### 1. Naive Bayes
A probabilistic classifier based on Bayes theorem that assumes independence between words.

**Pros**
- Very fast
- Works well on small datasets
- Strong baseline for NLP

**Cons**
- Cannot capture context

---

### 2. Support Vector Machine (SVM)
Finds the optimal boundary separating sports and politics texts.

**Pros**
- High accuracy
- Handles high dimensional TF-IDF vectors well

**Cons**
- Slightly slower training

---

### 3. Logistic Regression
Predicts probability of category using sigmoid function.

**Pros**
- Interpretable
- Simple model

**Cons**
- Needs larger dataset for better performance

---

## How to Run

### Install Dependency
```bash
pip install scikit-learn
