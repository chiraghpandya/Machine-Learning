import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#read CSV file
iris=pd.read_csv("D:/4070/today/iris.csv")

#features
x = iris[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

#Target
y=iris["Species"]

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2,random_state=50
)

#Create Decision Tree model
model=DecisionTreeClassifier(max_depth=3,random_state=1)

#Train model
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

#Accuracy
print("Accuracy:", accuracy_score(y_test,y_pred))

#confusion matrix
print("\nconfusion matrix:")
print(confusion_matrix(y_test,y_pred))

#classification Report
print("\n Classification Report:")
print(classification_report(y_test,y_pred))

