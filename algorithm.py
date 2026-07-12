import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)   
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


lr=LinearRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)
print("Prediction:",y_pred)

print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

lg=LogisticRegression()
lg.fit(X_train,y_train)

y_pred=lg.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("Classification Report: \n",classification_report(y_test,y_pred))

# Decision Tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dc=DecisionTreeClassifier()
dc.fit(X_train,y_train)

y_pred=dc.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rf=RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))

# K-Nearest Neighbors 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
print("Classification Report: \n",classification_report(y_test,y_pred))

# Support Vector Machine (SVM)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sv=SVC()
sv.fit(X_train,y_train)

y_pred=sv.predict(X_test)
print("Accuracy: ",accuracy_score(y_test,y_pred))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

nv = GaussianNB()

nv.fit(X_train, y_train)

y_pred= nv.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

gd = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gd.fit(X_train, y_train)

y_pred = gd.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#AdaBoost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report


model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train, y_train)

y_pred= model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Kmeans Clustering

from sklearn.cluster import KMeans
km=KMeans(n_clusters=2,random_state=42)

cluster_labels=km.fit_predict(X)
print("Cluster Labels:", cluster_labels)

# DBSCAN

from sklearn.cluster import DBSCAN
db=DBSCAN(eps=3,min_samples=2)
cluster=db.fit_predict(X)

print("DBSCAN Cluster Labels:", cluster)

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=2)
cluster_labels = hc.fit_predict(X)
print("Hierarchical Cluster Labels:", cluster_labels)

# PCA (Principal Component Analysis)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_pca=pca.fit_transform(X)

print("PCA Transformed Data:\n", x_pca)

# Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

gs = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators":[50,100],
    "max_depth":[3,5,None]
}

grid = GridSearchCV(gs, param_grid, cv=5)

grid.fit(X,y)

print(grid.best_params_)
print(grid.best_score_)

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

cl = DecisionTreeClassifier()

scores = cross_val_score(cl, X, y, cv=5)

print(scores)
print("Average Accuracy:", scores.mean())

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
prob = lg.predict_proba(X_test)[:,1]

precision, recall, threshold = precision_recall_curve(y_test, prob)

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.show()

#StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

#MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

min=MinMaxScaler()
X_minmax = min.fit_transform(X)
print(X_minmax[:5])

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
X_onehot = ohe.fit_transform(X)
print(X_onehot[:5])