import streamlit as st
import pandas as pd
import numpy as np
from tree.decision_tree import DecisionTree
from tree.criteria import GiniCriterion, EntropyCriterion, MSECriterion
from preprocessing.feature_types import determine_type_of_feature
from preprocessing.filter import filter_data
from preprocessing.split import train_test_split
from visualization.visualize import visualize_tree
from tree.hyperparameter_tuning import tune_hyperparameters
import graphviz
import os
# Removed sklearn.metrics imports
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, mean_absolute_error

# Import custom evaluation metrics and plotting functions
from evaluation.classification import calculate_accuracy, plot_confusion_matrix
from evaluation.regression import calculate_r_squared, calculate_mean_absolute_error, create_plot
import seaborn as sns # Keep for feature importance plot, if re-enabled
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="🌳 Decision Tree Implementation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Make sidebar slimmer and darker */
    section[data-testid="stSidebar"] {
        width: 220px !important;
        background-color: #1E293B !important;
    }

    /* Sidebar content styling */
    .css-1d391kg .sidebar-content {
        background-color: #1E293B !important; /* Ensure content area matches sidebar */
        padding: 1rem;
    }

    .css-1d391kg .sidebar-nav {
        background-color: #1E293B !important;
    }

    .css-1d391kg .sidebar-nav a {
        color: #E0F2FE !important;  /* Light text for contrast */
        font-weight: 500;
        padding: 0.4rem 1rem;
        display: block;
        margin: 0.3rem 0;
        text-decoration: none;
        font-size: 0.95rem;
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 4px; /* Ensure hover background has rounded corners */
    }

    .css-1d391kg .sidebar-nav a:hover {
        background-color: #2563EB !important;
        color: #ffffff !important;
    }

    .css-1d391kg .sidebar-nav a.active {
        background-color: #2563EB !important;
        color: #ffffff !important;
    }

    /* Targeting Streamlit radio buttons specifically within the sidebar */
    /* These rules ensure the radio buttons in the sidebar have light text and no white box */
    div[data-testid="stRadio"],
    div[data-testid="stRadio"] > div,
    div[data-testid="stRadio"] > label,
    div[data-testid="stRadio"] > label > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0px !important; /* Remove any default margins */
        padding: 0px !important; /* Ensure no padding is causing a background */
    }

    div[data-testid="stRadio"] > label {
        padding: 0.4rem 1rem !important; /* Restore padding for clickable area */
        color: #E0F2FE !important; /* Ensure label text is light */
    }

    div[data-testid="stRadio"] > label:hover {
        background-color: #2563EB !important; /* Blue background on hover */
        border-radius: 4px; /* Apply border-radius on hover for the label */
    }

    div[data-testid="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        color: #E0F2FE !important; /* Ensure markdown text within label is light */
    }

    /* Hide the actual radio input element and its dot */
    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stRadio"] > label > div:first-child {
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
        background: transparent !important;
        border: none !important;
        outline: none !important;
        width: 0px !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        visibility: hidden !important;
    }

    /* General layout - Black background, white text */
    .stApp {
        max-width: 1600px;
        margin: 0 auto;
        background-color: #000000; /* Black background for the entire app */
    }

    .main {
        background-color: #000000; /* Black background for main content area */
        padding: 2rem;
        color: #ffffff; /* White text for main content */
    }

    /* Titles */
    .title-text {
        font-size: 3.2em; /* Adjusted to a more appropriate, balanced size */
        color: #4DD0E1 !important; /* Vibrant blue for the main title */
        text-align: center;
        margin-bottom: 1em;
        font-weight: bold;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }

    .subtitle-text {
        font-size: 2.2em; /* Adjusted subtitle size */
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 2em;
        font-weight: 500;
    }

    .section-header {
        font-size: 2.5em; /* Adjusted section header size */
        color: #ffffff !important;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
        font-weight: bold;
        border-bottom: 2px solid #555555; /* Adjusted border color for dark bg */
        padding-bottom: 0.5em;
    }

    /* Content boxes - adjust for black background */
    .info-box,
    .feature-box {
        background-color: #1A1A1A; /* Darker gray for boxes on black background */
        color: #ffffff; /* White text inside boxes */
        border: 1px solid #333333; /* Darker border */
        box-shadow: 0 2px 4px rgba(255,255,255,0.05); /* Lighter shadow for dark bg */
    }

    /* Text styling - ensure all main text is white */
    .stMarkdown,
    .stText,
    .stCode,
    .stJson,
    p,
    span,
    div:not([data-testid="stSidebar"]) div,
    h1, h2, h3, h4, h5, h6,
    /* Ensure Streamlit's internal text containers are white */
    .st-emotion-cache-1c7y2qn,
    .st-emotion-cache-lnq447,
    .st-emotion-cache-1jm6hpb,
    .st-emotion-cache-10o5p32,
    .st-emotion-cache-fx762l,
    .st-emotion-cache-12m5y0i,
    .st-emotion-cache-snadqp,
    .st-emotion-cache-1fv4yvx
    {
        color: #ffffff !important;
    }

    /* DataFrames - adjust for black background */
    .stDataFrame {
        background-color: #1A1A1A; /* Darker gray background for DataFrames */
        color: #ffffff; /* White text for DataFrame content */
        border: 1px solid #333333; /* Darker border */
        box-shadow: 0 2px 4px rgba(255,255,255,0.05);
    }

    .stDataFrame th {
        color: #ffffff !important; /* White text for DataFrame headers */
    }

    /* Form elements - adjust for black background */
    .stSelectbox, .stRadio, .stNumberInput, .stFileUploader, .stTextInput, .stTextArea, .stDateInput, .stTimeInput, .stColorPicker, .stSlider, .stCheckbox {
        background-color: #1A1A1A; /* Darker gray background for form elements */
        color: #ffffff; /* White text for input values */
        border: 1px solid #333333;
        box-shadow: 0 1px 3px rgba(255, 255, 255, 0.05);
    }

    /* Form labels */
    .stSelectbox label,
    .stRadio label,
    .stNumberInput label,
    .stFileUploader label,
    .stTextInput label,
    .stTextArea label,
    .stDateInput label,
    .stTimeInput label,
    .stColorPicker label,
    .stSlider label,
    .stCheckbox label,
    .stButton label {
        color: #ffffff !important; /* White text for all form labels */
    }

    /* Input field text and placeholders */
    .stTextInput div[data-testid="stTextInput"] > div > input,
    .stNumberInput div[data-testid="stNumberInput"] > div > input,
    .stTextArea div[data-testid="stTextArea"] > div > textarea,
    input::placeholder,
    textarea::placeholder {
        color: #E0E0E0 !important; /* Lighter white for placeholders */
    }

    /* File uploader button text and description */
    .stFileUploader section[data-testid="stFileUploader"] > div > button,
    .stFileUploader section[data-testid="stFileUploader"] > div > div > button,
    .stFileUploader section[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p,
    .stFileUploader section[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] {
        color: #ffffff !important; /* White text for file uploader buttons and description */
    }

    /* Buttons */
    .stButton button {
        background-color: #2563EB; /* Keep original button color */
        color: white;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
    }

    /* Error messages and warnings - adjust for black background */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid;
    }

    .stAlert[data-baseweb="notification"] {
        background-color: #330000 !important; /* Dark red background for errors */
        border-color: #FF6666 !important;
        color: #FFDDDD !important; /* Light red text for errors */
    }

    .stAlert[data-baseweb="notification"].error {
        background-color: #330000 !important;
        border-color: #FF6666 !important;
        color: #FFDDDD !important;
    }

    .stAlert[data-baseweb="notification"].warning {
        background-color: #332200 !important; /* Dark orange background for warnings */
        border-color: #FFCC66 !important;
        color: #FFF2DD !important;
    }

    .stAlert[data-baseweb="notification"].info {
        background-color: #000033 !important; /* Dark blue background for info */
        border-color: #6666FF !important;
        color: #DDDDFF !important;
    }

    .stAlert[data-baseweb="notification"].success {
        background-color: #003300 !important; /* Dark green background for success */
        border-color: #66FF66 !important;
        color: #DDFFDD !important;
    }

    /* Code blocks and technical content - adjust for black background */
    .stCodeBlock {
        background-color: #1A1A1A !important; /* Darker gray background */
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    .stCodeBlock pre {
        background-color: #1A1A1A !important;
        color: #ffffff !important;
    }

    /* Scrollbar styling - adjust for dark background */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1A1A1A;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb {
        background: #555555;
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #777777;
    }

    /* Metrics and statistics - adjust for black background */
    .stMetric {
        background-color: #1A1A1A; /* Darker gray background */
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .stMetric label {
        color: #CCCCCC; /* Light gray for labels */
        font-size: 0.9rem;
    }

    .stMetric div {
        color: #E0E0E0; /* Lighter white for values */
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)



def show_home_page():
    st.markdown('<div class="title-text">🌳 DECISION TREE IMPLEMENTATION FROM SCRATCH</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="padding: 1.5rem; border-radius: 8px;">
    <h3 style="color: #4DD0E1; font-size: 1.8em; margin-bottom: 1rem;">👋 Welcome to Our Custom Decision Tree Builder!</h3>
    <p style="font-size: 1.1em; line-height: 1.6;">
    This interactive web application allows you to build, visualize, and analyze 🌳 Decision Trees from your own datasets. Our core decision tree algorithm is built entirely from scratch, providing a transparent and educational experience.
    </p>

    <h4 style="color: #64B5F6; margin-top: 1.5rem; margin-bottom: 0.8rem;">✨ Key Capabilities:</h4>
    <ul style="font-size: 1.05em; line-height: 1.8;">
        <li><p>🚀 <strong>End-to-End Workflow:</strong> Upload your dataset, preprocess it, build the tree, visualize it, and evaluate its performance, all within a few clicks!</p></li>
        <li><p>🎯 <strong>Classification & Regression:</strong> Automatically detects your problem type (classification or regression) based on your target variable and applies the appropriate impurity criteria (Gini/Entropy for Classification, MSE for Regression).</p></li>
        <li><p>🧹 <strong>Intelligent Preprocessing:</strong>
            <ul>
                <li>➖ Handles missing values with strategies like dropping rows, or filling with mean, median, or mode.</li>
                <li>🔢 Automatically encodes categorical features to numerical representations.</li>
                <li>🚫 Smartly detects and suggests exclusion of ID columns.</li>
                <li>🗑️ <strong>NEW! Explicit Feature Exclusion:</strong> You can now manually select and remove any columns you deem irrelevant (e.g., 'Name' column in Titanic dataset) to refine your feature set for better model performance.</li>
            </ul>
        </p></li>
        <li><p>⚙️ <strong>NEW! Hyperparameter Tuning (Grid Search with K-Fold CV):</strong> Optimize your tree's performance! Define ranges for key hyperparameters like `Max Depth` and `Minimum Samples for Split`. Our system uses K-Fold Cross-Validation to find the best combination for your data.</p></li>
        <li><p>📊 <strong>NEW! Feature Importance:</strong> Gain insights into your data! Our tree calculates and displays the importance of each feature, showing which ones contribute most significantly to the model's predictions.</p></li>
        <li><p>📈 <strong>Comprehensive Evaluation & Visualization:</strong>
            <ul>
                <li>Summary of model performance metrics (Accuracy, Classification Report for classification; MSE, MAE, R² for regression).</li>
                <li>Confusion Matrix for classification problems.</li>
                <li>📉 <strong>NEW! Regression Time Analysis:</strong> Visualize actual vs. predicted values over the sample index, helping you understand the model's performance pattern.</li>
                <li>Interactive 🌲 tree visualization using Graphviz.</li>
            </ul>
        </p></li>
    </ul>

    <h4 style="color: #64B5F6; margin-top: 1.5rem; margin-bottom: 0.8rem;">🛠️ Our Custom Core vs. Standard Libraries:</h4>
    <p style="font-size: 1.05em; line-height: 1.6;">
    Rest assured, the heart of this project—including the **decision tree algorithm's construction, splitting logic, criterion calculations, and all data preprocessing steps (including train-test splitting)**—is entirely custom-built. ✨
    </p>
    <p style="font-size: 1.05em; line-height: 1.6;">
    We leverage popular libraries like `scikit-learn` for **standard model evaluation metrics** (e.g., accuracy, MSE) and `matplotlib`, `seaborn`, and `graphviz` for **visualization purposes only**. This ensures the validity and educational value of our custom implementation while providing robust, industry-standard performance assessment and powerful plotting capabilities.
    </p>

    <h4 style="color: #64B5F6; margin-top: 1.5rem; margin-bottom: 0.8rem;">🛠️ Our Custom Core vs. Standard Libraries:</h4>
    <p style="font-size: 1.05em; line-height: 1.6;">
    Rest assured, the heart of this project—including the **decision tree algorithm's construction, splitting logic, criterion calculations, and all data preprocessing steps (including train-test splitting)**—is entirely custom-built. ✨
    </p>
    <p style="font-size: 1.05em; line-height: 1.6;">
    We leverage popular libraries like `scikit-learn` for **standard model evaluation metrics** (e.g., accuracy, MSE) and `matplotlib`, `seaborn`, and `graphviz` for **visualization purposes only**. This ensures the validity and educational value of our custom implementation while providing robust, industry-standard performance assessment and powerful plotting capabilities.
    </p>

    <h4 style="color: #64B5F6; margin-top: 1.5rem; margin-bottom: 0.8rem;">🔬 A Closer Look at Our Custom Functions:</h4>
    <ul style="font-size: 1.05em; line-height: 1.8;">
        <li><p><code><strong>DecisionTree</strong></code> <strong>Class (</strong><code style="color: #E0F2FE;">tree/decision_tree.py</code><strong>):</strong> This is the main orchestrator of our custom decision tree. It encapsulates the entire model, managing the fitting process, making predictions, and calculating performance scores using our custom core logic.</p></li>
        <li><p><code><strong>decision_tree_algorithm</strong></code> <strong>(</strong><code style="color: #E0F2FE;">tree/tree_builder.py</code><strong>):</strong> The very heart of our tree! This recursive function intelligently builds the tree structure by deciding optimal split points at each node. It also powers our <strong style="color: #4DD0E1;">Feature Importance</strong> calculation by tracking impurity reductions.</p></li>
        <li><p><strong>Criterion Classes (</strong><code><strong>GiniCriterion</strong></code><strong>,</strong> <code><strong>EntropyCriterion</strong></code><strong>,</strong> <code><strong>MSECriterion</strong></code> <strong>in</strong> <code style="color: #E0F2FE;">tree/criteria.py</code><strong>):</strong> These classes define how our tree measures 'impurity' (e.g., Gini, Entropy for classification; Mean Squared Error for regression) and how the weighted average impurity of a split is calculated to find the best data divisions.</p></li>
        <li><p><strong>Split Functions (</strong><code><strong>get_potential_splits</strong></code><strong>,</strong> <code><strong>determine_best_split</strong></code><strong>,</strong> <code><strong>split_data</strong></code> <strong>in</strong> <code style="color: #E0F2FE;">tree/splits.py</code><strong>):</strong> These work in concert to find all possible data split points, identify the single best split based on impurity reduction, and physically divide the dataset accordingly.</p></li>
        <li><p><code><strong>predict_example</strong></code> <strong>(in</strong> <code style="color: #E0F2FE;">tree/predict.py</code><strong>):</strong> A dedicated utility that navigates our custom-built tree structure for a single input data point, following the learned rules to produce a prediction.</p></li>
        <li><p><strong>Helper Functions (e.g.,</strong> <code><strong>create_leaf</strong></code><strong>,</strong> <code><strong>check_purity</strong></code> <strong>in</strong> <code style="color: #E0F2FE;">tree/helper.py</code><strong>):</strong> Foundational utilities that handle tasks like creating the tree's final prediction nodes (leaves) and efficiently determining if a data segment is 'pure' enough to stop further tree growth.</p></li>
        <li><p><strong>Preprocessing Functions (e.g.,</strong> <code><strong>determine_type_of_feature</strong></code> <strong>in</strong> <code style="color: #E0F2FE;">preprocessing/feature_types.py</code><strong>,</strong> <code><strong>filter_data</strong></code> <strong>in</strong> <code style="color: #E0F2FE;">preprocessing/filter.py</code><strong>):</strong> Our custom implementations for initial data preparation, including identifying feature types and handling missing values.</p></li>
        <li><p><code><strong>train_test_split</strong></code> <strong>(in</strong> <code style="color: #E0F2FE;">preprocessing/split.py</code><strong>):</strong> Our own custom function for dividing your dataset into training and testing subsets, ensuring full control over this crucial data preparation step.</p></li>
        <li><p><code><strong>visualize_tree</strong></code> <strong>(in</strong> <code style="color: #E0F2FE;">visualization/visualize.py</code><strong>):</strong> This function is specifically designed to convert our custom tree structure into a compatible format for Graphviz, enabling the interactive and visual display of your decision tree.</p></li>
    </ul>

    <h4 style="color: #64B5F6; margin-top: 1.5rem; margin-bottom: 0.8rem;">🚀 Get Started:</h4>
    <p style="font-size: 1.05em; line-height: 1.6;">
    Navigate to the <strong>"Build Decision Tree"</strong> tab in the sidebar, upload your CSV dataset, and start exploring!
    </p>
    </div>
    """
    , unsafe_allow_html=True)

def show_data_overview(data):
    st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Information")
        st.write(f"Number of rows: {data.shape[0]}")
        st.write(f"Number of columns: {data.shape[1]}")
        
        # Display data types
        st.markdown("### Data Types")
        st.write(data.dtypes)
    
    with col2:
        st.markdown("### Missing Values")
        missing_values = data.isnull().sum()
        st.write(missing_values[missing_values > 0])
        
        if missing_values.sum() > 0:
            st.markdown('<div class="info-box">⚠️ Missing values detected. Consider preprocessing steps.</div>', unsafe_allow_html=True)
    
    # Data Preview
    st.markdown("### Data Preview")
    st.dataframe(data.head())
    
    # Basic Statistics
    st.markdown("### Basic Statistics")
    st.write(data.describe())

def show_preprocessing(data):
    st.markdown('<div class="section-header">Preprocessing Steps</div>', unsafe_allow_html=True)
    
    # Feature type detection (kept as it's used internally)
    feature_types = determine_type_of_feature(data)
    
    # Missing value handling
    st.markdown("### Missing Value Handling")
    missing_strategy = st.selectbox(
        "Select strategy for missing values",
        ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
        key="missing_strategy_selectbox"
    )

    # Handle categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.markdown("### Categorical Columns")
        st.info(f"Found categorical columns: {', '.join(categorical_cols)}")
        st.info("These will be automatically encoded for the decision tree.")

    return {
        "missing_strategy": missing_strategy,
        "scaling_method": "None",  # Defaulting to None as scaling is removed from UI
        "feature_selection": False,  # Defaulting to False as feature selection is removed from UI
        "feature_threshold": None
    }

def show_tree_building(y):
    st.markdown('<div class="section-header">Tree Building Parameters</div>', unsafe_allow_html=True)
    
    # Automatically determine problem type and criterion based on target variable
    if pd.api.types.is_numeric_dtype(y):
        if len(y.unique()) < 0.2 * len(y) and len(y.unique()) <= 20: # Heuristic for classification vs regression
            ml_task = "classification"
            st.markdown("Detected Problem Type: **Classification**")
            criterion_obj = GiniCriterion() # Default to Gini for simplicity
            st.markdown("Criterion: **Gini Impurity** (default for classification)")
        else:
            ml_task = "regression"
            st.markdown("Detected Problem Type: **Regression**")
            criterion_obj = MSECriterion()
            st.markdown("Criterion: **Mean Squared Error** (default for regression)")
    else:
        ml_task = "classification"
        st.markdown("Detected Problem Type: **Classification**")
        criterion_obj = GiniCriterion() # Default to Gini for non-numeric classification
        st.markdown("Criterion: **Gini Impurity** (default for classification)")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Tree Parameters")
        max_depth = st.number_input("Maximum tree depth", min_value=1, max_value=20, value=5, key="max_depth_input")
    
    with col2:
        min_samples_split = st.number_input("Minimum samples for split", min_value=2, value=2, key="min_samples_split_input")
    
    return {
        "problem_type": ml_task,
        "criterion": criterion_obj,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": 1  # Set to default value of 1
    }

def show_metrics(X_test, y_test, y_pred, tree, problem_type):
    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if problem_type == "classification":
            st.markdown("### Classification Metrics")
            if not isinstance(y_test, pd.Series):
                y_true_series = pd.Series(y_test, name='label')
            else:
                y_true_series = y_test.rename('label')
            
            # For custom calculate_accuracy, we need a DataFrame containing features (X_test) and the true labels (y_true_series).
            # The custom calculate_accuracy also performs predictions internally using the passed tree.
            # So, we pass X_test and y_true_series combined.
            test_df_for_metrics = pd.concat([X_test.reset_index(drop=True), y_true_series.reset_index(drop=True)], axis=1)

            accuracy = calculate_accuracy(test_df_for_metrics, tree.tree) # Pass tree.tree (the dictionary) to custom func
            st.metric("Accuracy", f"{accuracy:.2f}")
            st.markdown("### Classification Report")
            from sklearn.metrics import classification_report # Keep for the full report output
            st.text(classification_report(y_test, y_pred))
        else:
            st.markdown("### Regression Metrics")
            test_df_for_metrics = pd.concat([X_test.reset_index(drop=True), y_test.rename('label').reset_index(drop=True)], axis=1)
            
            r2_custom = calculate_r_squared(test_df_for_metrics, tree.tree)
            mae_custom = calculate_mean_absolute_error(test_df_for_metrics, tree.tree)
            
            # For MSE, we use sklearn's metric as we don't have a custom one yet in evaluation/regression.py for this specific calculation
            # Update: calculate_mse is in tree/criteria.py, but it's for impurity, not directly for final model evaluation.
            # Keeping sklearn MSE for consistency with common usage or will need a custom one for final metric.
            from sklearn.metrics import mean_squared_error # Re-import specifically for MSE
            mse = mean_squared_error(y_test, y_pred)
            
            st.metric("Mean Squared Error", f"{mse:.2f}")
            st.metric("Mean Absolute Error", f"{mae_custom:.2f}")
            st.metric("R² Score", f"{r2_custom:.2f}")
    
    with col2:
        st.markdown("### Actual vs Predicted")
        if problem_type == "classification":
            test_df_for_metrics = pd.concat([X_test.reset_index(drop=True), y_test.rename('label').reset_index(drop=True)], axis=1)
            fig = plot_confusion_matrix(test_df_for_metrics, tree.tree, title="Confusion Matrix")
            st.pyplot(fig)
        else:
            test_df_for_metrics = pd.concat([X_test.reset_index(drop=True), y_test.rename('label').reset_index(drop=True)], axis=1)
            fig = create_plot(test_df_for_metrics, tree.tree, title="Actual vs Predicted Values")
            st.pyplot(fig)

def is_id_column(series, is_first_column=False):
    """
    Check if a column is likely an ID column.
    For first column only:
    - If it's named 'id', 'index', or contains 'id' in its name
    - If it's a sequential number (1,2,3...)
    For other columns:
    - Only if explicitly named as an ID column
    """
    column_name = series.name.lower()
    
    # For first column, check if it's a sequential ID
    if is_first_column:
        if column_name in ['id', 'index', 'idx'] or 'id' in column_name:
            return True
        # Check if it's a sequential number
        try:
            values = pd.to_numeric(series)
            if values.is_monotonic_increasing and (values.diff().dropna() == 1).all():
                return True
        except:
            pass
    
    # For other columns, only check the name
    return column_name in ['id', 'index', 'idx'] or column_name.endswith('_id')

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Build Decision Tree"])

    if page == "Home":
        show_home_page()
    else:
        st.markdown('<div class="title-text">Build Your Decision Tree</div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Load data
                data = pd.read_csv(uploaded_file)
                
                # Show data overview
                show_data_overview(data)
                
                # Target column selection
                target_col = st.selectbox("Select target column", data.columns)

                # --- Feature Exclusion and Automatic ID Detection ---
                st.markdown('<div class="section-header">Feature Exclusion</div>', unsafe_allow_html=True)
                
                # Get all columns except the target for exclusion options
                cols_for_exclusion_options = [col for col in data.columns if col != target_col]

                excluded_cols = st.multiselect(
                    "Select columns to explicitly exclude (e.g., non-predictive text like 'Name')",
                    options=cols_for_exclusion_options,
                    default=[]
                )

                # Drop explicitly excluded columns
                if excluded_cols:
                    st.info(f"Explicitly excluded columns: {', '.join(excluded_cols)}")
                    data = data.drop(columns=excluded_cols)
                
                # Re-check if target column was accidentally excluded
                if target_col not in data.columns: # Check if target_col is still in data after exclusion
                    st.error("The selected target column was explicitly excluded. Please re-select a valid target column.")
                    st.stop() # Stop execution if target column is invalid

                # Now, perform automatic ID column detection on the potentially modified 'data'
                initial_data_columns = data.columns # Use current data columns for detection
                auto_detected_id_columns = []

                if len(initial_data_columns) > 0: # Ensure there are columns left
                    first_col_after_exclusion = initial_data_columns[0]
                    if is_id_column(data[first_col_after_exclusion], is_first_column=True) and first_col_after_exclusion != target_col:
                        auto_detected_id_columns.append(first_col_after_exclusion)
                    
                    # Check other columns for explicit ID names (only on remaining columns)
                    for col in initial_data_columns[1:]:
                        if is_id_column(data[col]) and col != target_col:
                            auto_detected_id_columns.append(col)

                if auto_detected_id_columns:
                    st.info(f"Automatically detected and excluded ID columns: {', '.join(auto_detected_id_columns)}")
                    data = data.drop(columns=auto_detected_id_columns)
                
                # --- Hyperparameter Tuning Section ---
                st.markdown('<div class="section-header">Hyperparameter Tuning</div>', unsafe_allow_html=True)
                enable_tuning = st.checkbox("Enable Hyperparameter Tuning (Grid Search)")

                tuned_max_depth_range = None
                tuned_min_samples_split_range = None

                if enable_tuning:
                    st.info("Define ranges for hyperparameters. K-Fold Cross-Validation will be used to find the best combination.")
                    col_ht1, col_ht2 = st.columns(2)
                    with col_ht1:
                        st.markdown("**Max Depth Range:**")
                        min_md = st.number_input("Min Max Depth", min_value=1, max_value=20, value=3, key="min_max_depth_ht")
                        max_md = st.number_input("Max Max Depth", min_value=min_md, max_value=20, value=7, key="max_max_depth_ht")
                        tuned_max_depth_range = list(range(min_md, max_md + 1))

                    with col_ht2:
                        st.markdown("**Min Samples Split Range:**")
                        min_mss = st.number_input("Min Samples Split", min_value=2, value=2, key="min_min_samples_split_ht")
                        max_mss = st.number_input("Max Samples Split", min_value=min_mss, value=10, key="max_min_samples_split_ht")
                        tuned_min_samples_split_range = list(range(min_mss, max_mss + 1))
                    
                    n_splits_cv = st.slider("Number of K-Fold Splits", min_value=2, max_value=10, value=5, key="n_splits_cv")

                
                # Show preprocessing options
                preprocessing_config = show_preprocessing(data)
                
                # Show tree building options (base parameters)
                # These parameters will be used if hyperparameter tuning is NOT enabled
                if not enable_tuning:
                    st.markdown('<div class="section-header">Manual Tree Parameters</div>', unsafe_allow_html=True)
                    tree_config = show_tree_building(data[target_col])
                else:
                    # Dummy tree config when tuning is enabled, actual params come from tuning
                    tree_config = {
                        "problem_type": "classification", # Will be updated by tuning
                        "criterion": GiniCriterion(),     # Will be updated by tuning
                        "max_depth": None, 
                        "min_samples_split": None,
                        "min_samples_leaf": 1
                    }
                
                if st.button("Build Decision Tree" if not enable_tuning else "Run Hyperparameter Tuning and Build Tree"):
                    with st.spinner("Preparing data..."):
                        # Preprocess data
                        X = data.drop(columns=[target_col])
                        y = data[target_col]
                        
                        # Handle categorical columns
                        categorical_cols = X.select_dtypes(include=['object']).columns
                        if len(categorical_cols) > 0:
                            # Convert categorical columns to numeric using label encoding
                            for col in categorical_cols:
                                X[col] = X[col].astype('category').cat.codes
                        
                        # Apply preprocessing
                        if preprocessing_config["missing_strategy"] == "Drop rows":
                            X = X.dropna()
                            y = y[X.index]
                        elif preprocessing_config["missing_strategy"] == "Fill with mean":
                            X = X.fillna(X.mean())
                        elif preprocessing_config["missing_strategy"] == "Fill with median":
                            X = X.fillna(X.median())
                        elif preprocessing_config["missing_strategy"] == "Fill with mode":
                            X = X.fillna(X.mode().iloc[0])

                        # Determine problem type and criterion based on target column if not set by tuning
                        if pd.api.types.is_numeric_dtype(y):
                            # If the target column has many unique values, assume regression
                            if y.nunique() > 20 and y.dtype != 'object': # Heuristic for regression
                                ml_task = "regression"
                                criterion_class = MSECriterion
                            else:
                                ml_task = "classification"
                                criterion_class = GiniCriterion # Default for numeric classification
                        else:
                            ml_task = "classification"
                            criterion_class = GiniCriterion # Default for non-numeric classification
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                    if enable_tuning:
                        st.write("Running Hyperparameter Tuning...")
                        param_grid = {}
                        if tuned_max_depth_range:
                            param_grid['max_depth'] = tuned_max_depth_range
                        if tuned_min_samples_split_range:
                            param_grid['min_samples_split'] = tuned_min_samples_split_range
                        
                        # If no ranges are selected, use default single values to prevent empty product
                        if not param_grid:
                            param_grid['max_depth'] = [5]
                            param_grid['min_samples_split'] = [2]

                        with st.spinner("Tuning hyperparameters..."):
                            best_model, best_params, best_score, tuning_results = tune_hyperparameters(
                                X_train, y_train, ml_task, criterion_class, param_grid, n_splits=n_splits_cv
                            )
                        
                        st.success("Hyperparameter tuning complete!")
                        st.markdown('<div class="section-header">Tuning Results</div>', unsafe_allow_html=True)
                        st.write("Best Parameters found:", best_params)
                        st.write(f"Best Cross-Validation Score: {best_score:.4f}")
                        
                        st.markdown("### All Tuning Combinations")
                        tuning_results_df = pd.DataFrame(tuning_results)
                        st.dataframe(tuning_results_df.sort_values(by="mean_score", ascending=False))

                        tree = best_model # Use the best model found by tuning
                        tree_config["problem_type"] = ml_task # Update problem type

                    else: # No tuning, build single tree
                        with st.spinner("Building the tree..."):
                            # Build tree with manually selected parameters
                            tree = DecisionTree(
                                criterion=tree_config["criterion"],
                                max_depth=tree_config["max_depth"],
                                min_samples_split=tree_config["min_samples_split"],
                                min_samples_leaf=tree_config["min_samples_leaf"]
                            )
                            tree.fit(X_train, y_train)
                    
                    # Make predictions on the test set from the chosen/tuned tree
                    y_pred = tree.predict(X_test)
                    
                    # Display Feature Importances
                    # st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
                    # if tree.feature_importances_ and any(tree.feature_importances_.values()): # Check if any importance is non-zero
                    #     importances_df = pd.DataFrame(
                    #         list(tree.feature_importances_.items()),
                    #         columns=['Feature', 'Importance']
                    #     )
                    #     importances_df = importances_df.sort_values(by='Importance', ascending=False)
                    #     st.dataframe(importances_df)

                    #     # Optional: Plot feature importances
                    #     fig_imp, ax_imp = plt.subplots()
                    #     sns.barplot(x='Importance', y='Feature', data=importances_df, ax=ax_imp)
                    #     ax_imp.set_title("Feature Importances")
                    #     st.pyplot(fig_imp)
                    # else:
                    #     st.info("Feature importances are not available or are all zero (e.g., if it's a single leaf node or tree couldn't split).")

                    # Visualize tree
                    st.markdown('<div class="section-header">Tree Visualization</div>', unsafe_allow_html=True)
                    if tree.tree is not None:  # Check if tree was built successfully
                        dot = visualize_tree(tree.tree)  # Pass tree.tree instead of tree object
                        if dot is not None:
                            st.graphviz_chart(dot)
                        else:
                            st.error("Failed to create tree visualization. Please ensure Graphviz is installed.")
                    else:
                        st.error("Tree was not built successfully.")
                    
                    # Show metrics
                    show_metrics(X_test, y_test, y_pred, tree, tree_config["problem_type"])

                    # Time analysis of actual vs predicted for regression problems
                    if tree_config["problem_type"] == "regression":
                        st.markdown('<div class="section-header">Regression Time Analysis (Actual vs Predicted)</div>', unsafe_allow_html=True)
                        
                        # Create a DataFrame for plotting
                        plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                        plot_df = plot_df.reset_index(drop=True) # Reset index for consistent plotting

                        fig_time, ax_time = plt.subplots(figsize=(10, 6))
                        ax_time.plot(plot_df.index, plot_df['Actual'], label='Actual Values', marker='o', markersize=4, linestyle='-')
                        ax_time.plot(plot_df.index, plot_df['Predicted'], label='Predicted Values', marker='x', markersize=4, linestyle='--')
                        ax_time.set_title('Actual vs Predicted Values Over Sample Index')
                        ax_time.set_xlabel('Sample Index')
                        ax_time.set_ylabel('Value')
                        ax_time.legend()
                        ax_time.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig_time)
                        st.info("This plot shows actual vs predicted values indexed by sample order. If your dataset has a time-series component, consider that the inherent order of rows in the uploaded CSV dictates the 'time' here.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please ensure your data is properly formatted and try again.")
                st.error("For the Titanic dataset, make sure to select 'Survived' as the target column.")

if __name__ == "__main__":
    main()
