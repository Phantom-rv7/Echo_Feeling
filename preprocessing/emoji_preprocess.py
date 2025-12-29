# emoji_preprocess.py
# -------------------
# Preprocess Emoji Sentiment dataset (CSV format)
# Uses a custom tokenizer so each emoji is treated as a token.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("data/emoji/Emoji_Sentiment_Data_v1.0.csv", encoding="utf-8")

print("First 5 rows of dataset:")
print(df.head())
print("Columns available:", df.columns)

# -------------------------------
# 2. Clean emoji column
# -------------------------------
df["emoji_clean"] = df["Emoji"].astype(str).str.strip()

# -------------------------------
# 3. Create sentiment labels
# -------------------------------
def dominant_sentiment(row):
    counts = {"negative": row["Negative"], "neutral": row["Neutral"], "positive": row["Positive"]}
    return max(counts, key=counts.get)

df["sentiment"] = df.apply(dominant_sentiment, axis=1)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# -------------------------------
# 4. Encode labels
# -------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(df["sentiment"])
print("\nLabel encoding mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# -------------------------------
# 5. Vectorize emojis with custom tokenizer
# -------------------------------
# Tokenizer that treats each character (emoji) as a token
def char_tokenizer(text):
    return list(text)

count_vectorizer = CountVectorizer(tokenizer=char_tokenizer, lowercase=False)
X_count = count_vectorizer.fit_transform(df["emoji_clean"])

tfidf_vectorizer = TfidfVectorizer(tokenizer=char_tokenizer, lowercase=False)
X_tfidf = tfidf_vectorizer.fit_transform(df["emoji_clean"])

print("\nVectorization complete.")
print("Bag-of-Words shape:", X_count.shape)
print("TF-IDF shape:", X_tfidf.shape)

# -------------------------------
# 6. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("\nTrain/Test split complete.")
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# -------------------------------
# Final summary
# -------------------------------
print("\n✅ Emoji preprocessing finished successfully!")
print("Dataset shape:", df.shape)
