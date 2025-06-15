import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def preprocess_csv(
    file_path,
    drop_threshold=0.5,
    fill_missing=True,
    encode_categorical=False,
    scale_numeric=False,
    scaler_type='standard',
    save_path=None
):
    """
    Preprocess a CSV file by handling missing values, encoding, and scaling.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file to be preprocessed.
    
    drop_threshold : float (default=0.5)
        Fraction of allowed missing values in columns. Columns with > this threshold will be dropped.

    fill_missing : bool (default=True)
        If True, fill remaining missing values. Otherwise, rows with nulls will be dropped.

    encode_categorical : bool (default=False)
        If True, encodes categorical columns using Label Encoding.

    scale_numeric : bool (default=False)
        If True, scales numeric columns. Options: 'standard' or 'minmax'.

    scaler_type : str (default='standard')
        Type of scaler to use: 'standard' or 'minmax'.

    save_path : str or None
        If provided, saves the cleaned dataset to this path as a new CSV.

    Returns:
    --------
    df : pd.DataFrame
        The cleaned DataFrame.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Drop columns with more than threshold null values
    null_frac = df.isnull().mean()
    df = df.loc[:, null_frac <= drop_threshold]

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    if fill_missing:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df.dropna(inplace=True)

    # Encode categorical variables if needed
    if encode_categorical:
        label_encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object', 'category']).columns:
            try:
                df[col] = label_encoder.fit_transform(df[col])
            except:
                pass  # skip columns that can't be encoded

    # Scale numeric columns
    if scale_numeric:
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save cleaned file
    if save_path:
        df.to_csv(save_path, index=False)

    return df
