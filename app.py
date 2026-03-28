import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# PAGE TITLE

st.title("🌸 Iris Flower Machine Learning Dashboard")


# LOAD DATASET

@st.cache_data
def load_data():
    df = pd.read_csv("iris.csv")
    df = df.drop("Id", axis=1)
    return df


df = load_data()


# SIDEBAR MENU

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "Visualization Dashboard",
        "Model Performance",
        "Feature Importance",
        "Predict Flower Species"
    ]
)


# DATASET OVERVIEW

if menu == "Dataset Overview":

    st.header("Dataset Overview")

    st.subheader("First 10 Rows")
    st.write(df.head(10))

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Statistical Summary")
    st.write(df.describe())


# VISUALIZATION DASHBOARD

elif menu == "Visualization Dashboard":

    st.header("📊 Interactive Visualization Dashboard")

    feature = st.selectbox(
        "Select Feature",
        df.columns[:-1]
    )


    st.subheader("Summary Statistics")
    st.write(df[feature].describe())


    st.subheader("Histogram")

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x=feature,
        hue="Species",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)


    st.subheader("Boxplot (Species Comparison)")

    fig, ax = plt.subplots()

    sns.boxplot(
        data=df,
        x="Species",
        y=feature,
        ax=ax
    )

    st.pyplot(fig)


    st.subheader("Violin Plot (Distribution Shape)")

    fig, ax = plt.subplots()

    sns.violinplot(
        data=df,
        x="Species",
        y=feature,
        ax=ax
    )

    st.pyplot(fig)


    # ⭐ ADD THIS SECTION HERE
    st.subheader("Species-wise Feature Comparison")

    avg_values = df.groupby("Species")[feature].mean()

    st.bar_chart(avg_values)


    # Optional Heatmap Toggle
    if st.checkbox("Show Correlation Heatmap"):

        fig, ax = plt.subplots(figsize=(6,4))

        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

# MODEL PERFORMANCE

elif menu == "Model Performance":

    st.header("Model Performance Comparison")

    X = df.drop("Species", axis=1)
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    log_model = LogisticRegression(max_iter=200)
    tree_model = DecisionTreeClassifier()
    knn_model = KNeighborsClassifier()


    log_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)


    log_pred = log_model.predict(X_test)
    tree_pred = tree_model.predict(X_test)
    knn_pred = knn_model.predict(X_test)


    st.write("Logistic Regression Accuracy:",
             accuracy_score(y_test, log_pred))

    st.write("Decision Tree Accuracy:",
             accuracy_score(y_test, tree_pred))

    st.write("KNN Accuracy:",
             accuracy_score(y_test, knn_pred))


    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, log_pred)

    fig, ax = plt.subplots()

    sns.heatmap(cm,
                annot=True,
                fmt="d",
                ax=ax)

    st.pyplot(fig)


    st.subheader("Classification Report")

    st.text(classification_report(y_test,
                                 log_pred))


# FEATURE IMPORTANCE

elif menu == "Feature Importance":

    st.header("Feature Importance")

    X = df.drop("Species", axis=1)
    y = df["Species"]

    model = DecisionTreeClassifier()

    model.fit(X, y)

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    st.bar_chart(importance_df.set_index("Feature"))


# PREDICTION PANEL

elif menu == "Predict Flower Species":

    st.header("Predict Flower Species")

    sepal_length = st.slider("Sepal Length", 4.0, 8.0)

    sepal_width = st.slider("Sepal Width", 2.0, 4.5)

    petal_length = st.slider("Petal Length", 1.0, 7.0)

    petal_width = st.slider("Petal Width", 0.1, 2.5)


    X = df.drop("Species", axis=1)
    y = df["Species"]

    model = LogisticRegression(max_iter=200)

    model.fit(X, y)


    if st.button("Predict Species"):

        prediction = model.predict([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])

        st.success(
            f"Predicted Species: {prediction[0]}"
        )