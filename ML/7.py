from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report

newsgroups=fetch_20newsgroups(subset='all')
x,y=newsgroups.data,newsgroups.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

nb_classifier=MultinomialNB()

vectorizer=TfidfVectorizer(stop_words='english',max_df=0.5)

x_train_tfidf=vectorizer.fit_transform(x_train)
x_test_tfidf=vectorizer.transform(x_test)

nb_classifier.fit(x_train_tfidf,y_train)
y_pred=nb_classifier.predict(x_test_tfidf)

accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred,target_names=newsgroups.target_names)
print("accuracy: "+str(accuracy))
print("classification report: "+str(report))
