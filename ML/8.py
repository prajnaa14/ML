import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

url="C:\\Users\\PRAJNA\\Downloads\\iris.csv"
column_names=['sepal_length','Sepal_width','Petal_length','Petal_width','class']
df=pd.read_csv(url,header=None,names=column_names)

x=df.drop('class',axis=1)
y=df['class']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)

print("accuracy: "+str(accuracy))
print("Classification: "+str(report))

