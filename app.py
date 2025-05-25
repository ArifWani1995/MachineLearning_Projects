import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="AI Tools - Machine Learning Explorer",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 AI Tools - Machine Learning Explorer")
st.markdown("""
    Explore various machine learning algorithms using built-in datasets.
    This tool helps you understand different ML concepts through interactive visualizations.
""")

# Sidebar
st.sidebar.header("Configuration")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer", "Wine", "Digits")
)

# Algorithm selection
classifier_name = st.sidebar.selectbox(
    "Select Algorithm",
    ("Random Forest", "KNN", "SVM")
)

# Load dataset
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_digits()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y, data.target_names

X, y, target_names = get_dataset(dataset_name)

# Display dataset info
st.write("## Dataset Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"Number of samples: {X.shape[0]}")
with col2:
    st.info(f"Number of features: {X.shape[1]}")
with col3:
    st.info(f"Number of classes: {len(np.unique(y))}")

# Data visualization
st.write("## Data Visualization")
viz_type = st.selectbox("Select Visualization", ["Feature Distribution", "Correlation Matrix", "PCA Plot"])

if viz_type == "Feature Distribution":
    feature = st.selectbox("Select Feature", X.columns)
    fig = px.histogram(X, x=feature, color=pd.Series(y).map(lambda x: target_names[x]),
                      title=f"Distribution of {feature}")
    st.plotly_chart(fig)

elif viz_type == "Correlation Matrix":
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title("Feature Correlation Matrix")
    st.pyplot(fig)

elif viz_type == "PCA Plot":
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Target'] = pd.Series(y).map(lambda x: target_names[x])
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target',
                    title='PCA Visualization')
    st.plotly_chart(fig)

# Model Training
st.write("## Model Training")
test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
def train_model(classifier_name, X_train, y_train):
    if classifier_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', random_state=42)
    
    model.fit(X_train, y_train)
    return model

if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        model = train_model(classifier_name, X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Display results
        st.write("### Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Classification Report:")
            report = classification_report(y_test, y_pred, target_names=target_names)
            st.code(report)
        
        with col2:
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_names,
                       yticklabels=target_names)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(fig)
        
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Accuracy: {accuracy:.2%}") 