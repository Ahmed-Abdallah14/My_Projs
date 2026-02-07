import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


data = {
    'text': [
        'Free money now!', 'Hello, how are you?',
        'Win a prize', 'Exclusive offer for you', 'Can we go to the cafe?'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)


cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']


model = MultinomialNB()
model.fit(X, y)


n_email = ["I love programming"]
n_email_vector = cv.transform(n_email)
prediction = model.predict(n_email_vector)

print(f"The email is classified as: {prediction[0]}")
