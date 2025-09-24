import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB



messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])


nltk.download('stopwords')

ps = PorterStemmer()

corpus = []


for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()


y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#  Naive Bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# 
y_pred = spam_detect_model.predict(X_test)

# Function
def check_spam(email):
    # Preprocess the input email
    review = re.sub('[^a-zA-Z]', ' ', email)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)

    # 
    email_vector = cv.transform([review]).toarray()

    # Predict if it is spam (1) or ham (0)
    prediction = spam_detect_model.predict(email_vector)
    
    if prediction[0] == 1:
        return "This is a spam email."
    else:
        return "This is not a spam email."

# Take email input from the user
user_email = input("Paste your email here to check if it is spam or not: ")

# Check if the email is spam or not
result = check_spam(user_email)
print(result)
