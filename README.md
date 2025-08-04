
# 📝 Hotel Review Sentiment Analysis

This project analyzes hotel reviews and classifies them as **Good** or **Bad** using machine learning models. It uses natural language processing (NLP) techniques to clean the text data, extract features, and apply classifiers to predict review sentiment.

---

## 📂 Dataset Information

* **Source**: Scraped from Booking.com
* **Size**: 515,000 customer reviews from 1,493 luxury hotels across Europe
* **License**: Publicly available data (originally owned by Booking.com)

### Dataset Columns (selected for this project):

* `Positive_Review`: Positive text comments
* `Negative_Review`: Negative text comments
* `Reviewer_Score`: Reviewer score (used to infer sentiment)
* `Hotel_Name`, `Reviewer_Nationality`, etc. (other features available, not used in current model)

For this project, we combine positive and negative reviews (or you can define sentiment based on reviewer score) into a single `review` column and use a `label` column as:

* `1` → Good Review
* `0` → Bad Review

---

## 🧠 Model Overview

### Steps:

1. **Text Cleaning**

   * Lowercasing
   * Removing punctuation, numbers, and URLs
   * Tokenization
   * Stopword removal
   * Stemming

2. **Feature Extraction**

   * TF-IDF Vectorization

3. **Model Training**

   * Logistic Regression
   * Random Forest
   * KNN (optional)

4. **Evaluation**

   * Accuracy Score
   * Classification Report

5. **Model Selection**

   * The best-performing model is saved for later use

---

## 🔍 Example Usage

```python
import joblib
model = joblib.load('best_review_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_review(text):
    cleaned = clean_text(text)  # use same preprocessing
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "Good Review" if pred == 1 else "Bad Review"

predict_review("The hotel was clean and staff were very friendly.")
```

---

## 📦 Dependencies

Make sure to install the following packages:

```bash
pip install pandas scikit-learn nltk joblib
```

Also, download necessary NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---


---

## 📈 Results

Each model is evaluated and compared. The best model is automatically selected based on accuracy:

Example Output:

```
Model Evaluation Results:

Logistic Regression Accuracy: 0.9323
Random Forest Accuracy: 0.9316

✅ Best model saved: LogisticRegression with accuracy 0.9323
```

