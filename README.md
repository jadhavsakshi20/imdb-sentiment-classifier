# 🎬 IMDB Movie Review Sentiment Classifier

An end-to-end NLP project that classifies movie reviews as positive or negative using text preprocessing, TF-IDF vectorization, and classical machine learning models.

Built using Python, NLTK, and Scikit-learn, this project follows a complete NLP pipeline: raw text → cleaning & tokenization → feature extraction (Bag of Words & TF-IDF) → model training → evaluation → error analysis.

---

## 📌 Business Problem

Understanding customer sentiment at scale is a common need across e-commerce, entertainment, and product review platforms. Manually reading thousands of reviews isn't feasible — automated sentiment classification lets businesses quickly gauge audience reception and flag negative feedback trends.

This project answers:

- Can we accurately predict whether a review is positive or negative purely from its text?
- Which classical ML model performs better on text data — Logistic Regression or Naive Bayes?
- Where does the model fail, and what does that reveal about the limitations of word-frequency-based approaches?

---

## 📈 Results Summary

| **Metric** | **Value** |
|---|---|
| Total Reviews Analyzed | 50,000 |
| Class Balance | 25,000 positive / 25,000 negative |
| Feature Extraction | TF-IDF (5,000 features) |
| Best Model | Logistic Regression |
| Logistic Regression Accuracy | **88.91%** |
| Naive Bayes Accuracy | 85.14% |

---

## 💡 Skills Demonstrated

- Text Cleaning & Preprocessing
- Tokenization (NLTK)
- Stopword Removal
- Bag of Words & TF-IDF Vectorization
- Classification (Logistic Regression, Naive Bayes)
- Model Evaluation & Comparison
- Error Analysis
- Working with Google Colab & Google Drive integration

---

## 🛠️ Tech Stack

**Language:** Python
**NLP Libraries:** NLTK, Scikit-learn
**Data Libraries:** Pandas, NumPy
**Environment:** Google Colab
**Version Control:** Git, GitHub

---

## 📁 Repository Structure
mdb-sentiment-classifier/
│
├── IMDB_Sentiment_Classifier.ipynb # Full notebook: cleaning → TF-IDF → models → error analysis
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🔍 Methodology

### 1. Data Cleaning & Preprocessing
- Lowercased all text
- Removed HTML tags (`<br />`) present in the raw scraped reviews
- Removed punctuation and numbers using regex
- Removed English stopwords using NLTK's stopword list

### 2. Tokenization
- Used NLTK's `word_tokenize` to split cleaned reviews into individual word tokens
- Verified cleaning quality by checking the most frequent words in the dataset

### 3. Feature Extraction
- Converted text to numeric features using both **Bag of Words** (`CountVectorizer`) and **TF-IDF** (`TfidfVectorizer`), capped at 5,000 features
- TF-IDF was used for final model training, since it down-weights generic words (like "movie", "film") and up-weights distinctive, sentiment-carrying words

### 4. Model Training & Evaluation
- Trained Logistic Regression and Multinomial Naive Bayes on an 80/20 train-test split
- Compared accuracy between both models

### 5. Error Analysis
- Manually reviewed misclassified examples to understand model limitations

---

## 📊 Key Findings & Insights

- **Logistic Regression outperformed Naive Bayes** (88.91% vs 85.14%), likely because Naive Bayes assumes word independence, while Logistic Regression can weigh combinations of words together.
- **Error analysis revealed a consistent failure pattern:** the model misclassified mixed-sentiment reviews (mostly positive, with some criticism) as negative.
- **Root cause:** TF-IDF treats words independently and has no concept of negation or context — a phrase like *"wasn't disappointed"* gets partially read as negative due to the word "disappointed," even though the actual meaning is positive.
- This is a well-known limitation of classical Bag-of-Words/TF-IDF approaches compared to context-aware models like BERT, which read words in sequence and can capture negation and nuance.

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.9+
- Jupyter Notebook or Google Colab

### Setup & Execution

**Clone the repository:**
```bash
git clone https://github.com/jadhavsakshi20/imdb-sentiment-classifier.git
cd imdb-sentiment-classifier
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Open and run the notebook:**
```bash
jupyter notebook IMDB_Sentiment_Classifier.ipynb
```

(Or open directly in Google Colab and upload the IMDB dataset when prompted.)

---

## 🔮 Future Improvements

- Experiment with n-grams (bigrams/trigrams) to partially capture word context (e.g., "not good" as one unit)
- Try a transformer-based model (like BERT) to handle negation and nuanced sentiment better
- Deploy as a simple web app where users can paste a review and get a live prediction

---

## 👤 Author

**Sakshi Jadhav**
GitHub: [@jadhavsakshi20](https://github.com/jadhavsakshi20)

---

## 📜 License & Acknowledgments

**Dataset:** IMDB Dataset of 50K Movie Reviews (Kaggle)
**License:** MIT License — free to use, modify, and distribute for educational and portfolio purposes.
