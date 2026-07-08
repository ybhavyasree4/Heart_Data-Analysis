import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Data Analysis", layout="wide")

st.title("Heart Data Analysis")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("Display First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Display Last 5 Rows")
    st.dataframe(df.tail())

    st.header("Data View Options")

    option = st.selectbox(
        "Choose an Option",
        [
            "Display All Records",
            "Display Shape",
            "Display Columns",
            "Display Data Types",
            "Display Missing Values",
            "Display Duplicate Rows",
            "Display Describe",
            "Display Selected Columns"
        ]
    )

    if option == "Display All Records":
        st.write(df)

    elif option == "Display Shape":
        st.write(df.shape)

    elif option == "Display Columns":
        st.write(df.columns)

    elif option == "Display Data Types":
        st.write(df.dtypes)

    elif option == "Display Missing Values":
        st.write(df.isnull().sum())

    elif option == "Display Duplicate Rows":
        st.write(df[df.duplicated()])

    elif option == "Display Describe":
        st.write(df.describe())

    elif option == "Display Selected Columns":
        selected_cols = st.multiselect("Select Columns", df.columns)

        if selected_cols:
            st.write(df[selected_cols])

    # Visualization
    st.header("Visualization")

    graph = st.selectbox(
        "Choose Plot",
        [
            "Scatter Plot",
            "Histogram",
            "Bar Chart",
            "Box Plot",
            "Line Chart"
        ]
    )

    if graph == "Scatter Plot":
        x = st.selectbox("Choose X Column", df.columns)
        y = st.selectbox("Choose Y Column", df.columns)

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
        st.pyplot(fig)

    elif graph == "Histogram":
        col = st.selectbox("Choose Column", df.columns)

        fig, ax = plt.subplots()
        sns.histplot(df[col], ax=ax)
        st.pyplot(fig)

    elif graph == "Bar Chart":
        x = st.selectbox("Choose X Column", df.columns)
        y = st.selectbox("Choose Y Column", df.columns)

        fig, ax = plt.subplots()
        sns.barplot(data=df, x=x, y=y, ax=ax)
        st.pyplot(fig)

    elif graph == "Box Plot":
        col = st.selectbox("Choose Column", df.columns)

        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax)
        st.pyplot(fig)

    elif graph == "Line Chart":
        x = st.selectbox("Choose X Column", df.columns)
        y = st.selectbox("Choose Y Column", df.columns)

        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=x, y=y, ax=ax)
        st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")