import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris

iris_data=load_iris()
x=iris_data.data[:,2:]
y=iris_data.target
print("shape of x: " +str(x.shape)+ "\nShape of y: "+str(y.shape))

tree_clf=DecisionTreeClassifier(criterion='gini',max_depth=2,random_state=100)
clf=tree_clf.fit(x,y)

plt.figure(figsize=(10,10))
plt.title("decision tree")
plot_tree(clf,filled=True)
plt.show()

print(tree_clf.predict_proba([[5,1.5]]))
otp=tree_clf.predict([[5,1.5]])
predicted=iris_data.target_names[otp]
print("Predicted class : ",predicted[0])
