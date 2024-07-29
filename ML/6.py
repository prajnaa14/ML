from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import load_iris

iris_data=load_iris()
x=iris_data.data[:,2:]
y=iris_data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

df=pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['target']=iris_data.target_names[iris_data.target]
df.head()

rnd_clf=RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(x_train,y_train)

y_pred=rnd_clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

print("Actual: \n"+str(y_test))
print("Predicted: \n"+str(y_pred))
print("accuracy: "+str(accuracy))
