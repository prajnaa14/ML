import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

svm_classification=SVC(kernel='linear',random_state=42)
svm_classification.fit(x_train,y_train)

y_pred=svm_classification.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

print("accuracy: ",accuracy)

x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
y_min,y_max=x[:,1].min()-1,x[:,1].max()+1

xx,yy= np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
z=svm_classification.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.8)
plt.scatter(x[:,0],x[:,1],c=y,marker='o',edgecolors='k')
plt.title("SVM classifier decision boundary ")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()