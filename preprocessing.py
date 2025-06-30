import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def load_column_names():
    """Load column names from dictionary.txt"""
    with open("data/dictionary.txt", 'r') as f:
        lines = f.readlines()
    
    column_names = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and header lines
        if not line or line.startswith('DATA DICTIONARY') or line.startswith('Nr Name'):
            continue
        
        # Stop when we reach the label sections (L0:, L1:, etc.)
        if line.startswith('L') and ':' in line:
            break
            
        # Extract column names from numbered entries
        if line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                column_names.append(parts[1])
                
        # Stop when we have all 86 columns
        if len(column_names) >= 86:
            break
    
    # Ensure we have exactly 86 columns
    if len(column_names) != 86:
        raise ValueError(f"Expected 86 column names, but found {len(column_names)}")
    
    return column_names

def load_data():
    """Load training and test data with proper column names"""
    # Load column names from dictionary
    column_names = load_column_names()
    
    # Load training data
    train_df = pd.read_csv("data/ticdata2000.txt", sep='\t', header=None)
    train_df.columns = column_names
    
    # Load test data
    test_df = pd.read_csv("data/ticeval2000.txt", sep='\t', header=None)
    test_df.columns = column_names[:-1]  # test set doesn't have the target
    
    # Load test targets
    test_targets = pd.read_csv("data/tictgts2000.txt", header=None).squeeze()
    test_df["CARAVAN"] = test_targets
    
    return train_df, test_df

def get_feature_groups(column_names):
    """Categorize features according to README specifications"""
    # Socio-demographic features (columns 1-43)
    socio_demographic = column_names[:43]
    
    # Product ownership features (columns 44-85)
    # Split into contributions (P*) and number of policies (A*)
    product_contributions = [col for col in column_names[43:85] if col.startswith('P')]
    product_counts = [col for col in column_names[43:85] if col.startswith('A')]
    
    return {
        'socio_demographic': socio_demographic,
        'product_contributions': product_contributions,
        'product_counts': product_counts,
        'target': 'CARAVAN'
    }

def preprocess_data(train_df, test_df, use_pca=True, n_components=0.95):
    """
    Preprocess data focusing on socio-demographic features for clustering
    as specified in the README
    """
    # Get feature groups
    column_names = load_column_names()
    feature_groups = get_feature_groups(column_names)
    
    # Combine datasets for preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Extract socio-demographic features for clustering (README requirement)
    X_socio = combined_df[feature_groups['socio_demographic']]
    
    # Extract product ownership features for analysis
    X_products = combined_df[feature_groups['product_contributions'] + feature_groups['product_counts']]
    
    # Target variable
    y = combined_df[feature_groups['target']]
    
    # Handle categorical variables (though this dataset is mostly numeric)
    categorical_cols = X_socio.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        le = LabelEncoder()
        for col in categorical_cols:
            X_socio[col] = le.fit_transform(X_socio[col].astype(str))
    
    # Feature scaling for socio-demographic features
    scaler_socio = StandardScaler()
    X_socio_scaled = scaler_socio.fit_transform(X_socio)
    
    # Feature scaling for product features
    scaler_products = StandardScaler()
    X_products_scaled = scaler_products.fit_transform(X_products)
    
    # Apply PCA for dimensionality reduction if requested
    pca_socio = None
    if use_pca:
        pca_socio = PCA(n_components=n_components)
        X_socio_scaled = pca_socio.fit_transform(X_socio_scaled)
    
    # Split back into train and test
    train_size = len(train_df)
    X_train_socio = X_socio_scaled[:train_size]
    X_test_socio = X_socio_scaled[train_size:]
    X_train_products = X_products_scaled[:train_size]
    X_test_products = X_products_scaled[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    return {
        'X_train_socio': X_train_socio,
        'X_test_socio': X_test_socio,
        'X_train_products': X_train_products,
        'X_test_products': X_test_products,
        'y_train': y_train,
        'y_test': y_test,
        'socio_features': feature_groups['socio_demographic'],
        'product_features': feature_groups['product_contributions'] + feature_groups['product_counts'],
        'pca_socio': pca_socio,
        'scaler_socio': scaler_socio,
        'scaler_products': scaler_products,
        'train_df': train_df,
        'test_df': test_df
    }

if __name__ == "__main__":
    # Load and preprocess data
    train_df, test_df = load_data()
    data = preprocess_data(train_df, test_df)
    
    print(f"Socio-demographic features: {len(data['socio_features'])}")
    print(f"Product features: {len(data['product_features'])}")
    print(f"Training data shape: {data['X_train_socio'].shape}")
    if data['pca_socio']:
        print(f"PCA components: {data['pca_socio'].n_components_}")
        print(f"Explained variance ratio: {sum(data['pca_socio'].explained_variance_ratio_):.2f}")

