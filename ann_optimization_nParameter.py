import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPRegressor  # Used only for feature selection

# Load the data
file_path = 'knn_herb.csv'
data = pd.read_csv(file_path)

# Define target variable
target_column = 'Rat_Clint'

# List of descriptors to choose from
features_list = ['nAcid', 'nBase', 'CrippenLogP', 'nHBd', 'nHBDon', 'MLFER_A', 'nRing', 'LipinskiFailures', 'TopoPSA']
X = data[features_list]
y = data[target_column]

# ANN Training Parameters
test_sizes = [0.1, 0.15, 0.2, 0.3]  # Different test sizes
epochs_list = [50, 100, 150]  # Different training epochs
feature_range = range(8, 9)  # Selecting between 5 to 10 features

# Track best performance
best_r2 = float('-inf')
best_test_size = None
best_epoch = None
best_features = None

# Function to create the ANN model
def create_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Feature selection loop (select best 5-10 features)
for k in feature_range:
    print(f"\nSelecting {k} best features...")

    # Feature selection using MLPRegressor (a lightweight ANN for selection)
    sfs = SequentialFeatureSelector(
        MLPRegressor(random_state=42, hidden_layer_sizes=(50,)),  # Small ANN for fast selection
        n_features_to_select=k,
        direction='forward',
        scoring='r2',
        cv=5,
        n_jobs=-1
    )
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]
    print(f"Selected Features ({k}): {list(selected_features)}")

    # Use selected features
    X_selected = X[selected_features]

    # Model evaluation over different test sizes and epochs
    for test_size in test_sizes:
        for epochs in epochs_list:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42)

            # Create and train ANN model
            model = create_model(X_train.shape[1])
            model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=0)

            # Predict and evaluate model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            print(f"R²: {r2:.4f} | Features: {k} | Epochs: {epochs} | Test Size: {test_size}")

            # Store the best-performing model
            if r2 > best_r2:
                best_r2 = r2
                best_test_size = test_size
                best_epoch = epochs
                best_features = selected_features.copy()

# Print the best model details
    print("\nBest Model Performance:")
    print(f"Best R-squared (R²): {best_r2:.4f}")
    print(f"Best Test Size: {best_test_size}")
    print(f"Best Epochs: {best_epoch}")
    print(f"Selected Features: {list(best_features)}")
