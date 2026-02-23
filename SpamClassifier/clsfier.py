print("Program started...")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Loading dataset...")

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

print("Dataset loaded!")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

print("First 5 rows:")
print(data.head())

# Convert labels
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Vectorization
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label_num']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Program finished successfully.")