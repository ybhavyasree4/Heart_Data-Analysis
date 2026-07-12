import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_recall_curve

st.title("Heart Disease Prediction using Machine Learning")

uploaded_file = st.file_uploader("Upload Heart Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.25,random_state=42
    )

    option = st.sidebar.selectbox(
        "Choose Algorithm",
        (
            "Linear Regression",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "SVM",
            "Naive Bayes",
            "Gradient Boosting",
            "AdaBoost",
            "KMeans",
            "DBSCAN",
            "Hierarchical Clustering",
            "PCA",
            "Grid Search",
            "Cross Validation",
            "Precision Recall Curve",
            "StandardScaler",
            "MinMaxScaler",
            "OneHotEncoder"
        )
    )

    if st.button("Run"):

        if option=="Linear Regression":

            model=LinearRegression()
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.write("MSE:",mean_squared_error(y_test,pred))
            st.write("R2 Score:",r2_score(y_test,pred))

        elif option=="Logistic Regression":

            model=LogisticRegression(max_iter=1000)
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.write("Accuracy:",accuracy_score(y_test,pred))
            st.write(confusion_matrix(y_test,pred))
            st.text(classification_report(y_test,pred))

        elif option=="Decision Tree":

            model=DecisionTreeClassifier()
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.write("Accuracy:",accuracy_score(y_test,pred))

        elif option=="Random Forest":

            model=RandomForestClassifier()
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.write("Accuracy:",accuracy_score(y_test,pred))
            st.write(confusion_matrix(y_test,pred))

        elif option=="KNN":

            model=KNeighborsClassifier()
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.text(classification_report(y_test,pred))

        elif option=="SVM":

            model=SVC()
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.write("Accuracy:",accuracy_score(y_test,pred))

        elif option=="Naive Bayes":

            model=GaussianNB()
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.write("Accuracy:",accuracy_score(y_test,pred))

        elif option=="Gradient Boosting":

            model=GradientBoostingClassifier()
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.write("Accuracy:",accuracy_score(y_test,pred))

        elif option=="AdaBoost":

            model=AdaBoostClassifier()
            model.fit(X_train,y_train)

            pred=model.predict(X_test)

            st.write("Accuracy:",accuracy_score(y_test,pred))

        elif option=="KMeans":

            model=KMeans(n_clusters=2,random_state=42)

            cluster=model.fit_predict(X)

            st.write(cluster)

        elif option=="DBSCAN":

            model=DBSCAN(eps=3,min_samples=2)

            cluster=model.fit_predict(X)

            st.write(cluster)

        elif option=="Hierarchical Clustering":

            model=AgglomerativeClustering(n_clusters=2)

            cluster=model.fit_predict(X)

            st.write(cluster)

        elif option=="PCA":

            pca=PCA(n_components=2)

            result=pca.fit_transform(X)

            st.write(result)

        elif option=="Grid Search":

            model=RandomForestClassifier()

            param_grid={
                "n_estimators":[50,100],
                "max_depth":[3,5,None]
            }

            grid=GridSearchCV(model,param_grid,cv=5)

            grid.fit(X,y)

            st.write("Best Parameters")
            st.write(grid.best_params_)

            st.write("Best Score")
            st.write(grid.best_score_)

        elif option=="Cross Validation":

            model=DecisionTreeClassifier()

            score=cross_val_score(model,X,y,cv=5)

            st.write(score)
            st.write("Average Accuracy:",score.mean())

        elif option=="Precision Recall Curve":

            model=LogisticRegression(max_iter=1000)

            model.fit(X_train,y_train)

            prob=model.predict_proba(X_test)[:,1]

            precision,recall,_=precision_recall_curve(y_test,prob)

            fig,ax=plt.subplots()

            ax.plot(recall,precision)

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

            st.pyplot(fig)

        elif option=="StandardScaler":

            scaler=StandardScaler()

            scaled=scaler.fit_transform(X)

            st.write(scaled)

        elif option=="MinMaxScaler":

            scaler=MinMaxScaler()

            scaled=scaler.fit_transform(X)

            st.write(scaled)

        elif option=="OneHotEncoder":

            encoder=OneHotEncoder()

            encoded=encoder.fit_transform(X)

            st.write(encoded.toarray())