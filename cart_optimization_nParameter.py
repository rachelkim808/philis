import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector


file_path = 'knn_insect.csv'
data = pd.read_csv(file_path)

target_column = 'Rat_Clint'

features_list = [
    'nAcid', 'XLogP', 'apol', 'naAromAtom', 'nAromBond', 'nAtom', 'ATS0m', 'nBase',
    'BCUTw-1l', 'nBonds', 'bpol', 'SpMax1_Bhm', 'C1SP1', 'SCH-3', 'SC-3', 'SPC-4',
    'SP-0', 'Sv', 'CrippenLogP', 'SpMax_Dt', 'ECCEN', 'nHBd', 'ETA_Alpha', 'FMF', 'fragC',
    'nHBAcc', 'nHBDon', 'HybRatio', 'IC0', 'Kier1', 'nAtomLC', 'nAtomP', 'nAtomLAC',
    'MLogP', 'McGowan_Volume', 'MDEC-11', 'MLFER_A', 'MPC2', 'PetitjeanNumber', 'nRing',
    'nRotB', 'LipinskiFailures', 'topoRadius', 'GGI1', 'SpMax_D', 'TopoPSA', 'VABC',
    'MWC2', 'MW', 'WTPT-1', 'WPATH', 'XLogP', 'Zagreb'
]

X = data[features_list]
y = data[target_column]


test_sizes = [0.1, 0.2, 0.3]
random_states = range(0, 50)


best_overall_r2 = float('-inf')
best_overall_fold_error = None
best_overall_test_size = None
best_overall_random_state = None
best_overall_k = None
best_features = None

for k in range(7, 8):
    print(f"\nEvaluating feature combinations with {k} descriptors...")
    

    sfs = SequentialFeatureSelector(
        DecisionTreeRegressor(random_state=42),
        n_features_to_select=k,
        direction='forward',
        scoring='r2',
        cv=5,
        n_jobs=-1
    )
    sfs.fit(X, y)
    selected_features = X.columns[sfs.get_support()]
    print(f"Selected {k} features: {list(selected_features)}")
    
    X_selected = X[selected_features]
    
    for test_size in test_sizes:
        for random_state in random_states:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=random_state)
            
            model = DecisionTreeRegressor(random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)

            fold_error = np.mean(np.abs((y_pred / y_test) - 1))
            

            # print(f"k: {k}, Test_size: {test_size}, Random_state: {random_state}, R2: {r2:.4f}, AAFE: {fold_error:.4f}")
            

            if r2 > best_overall_r2:
                best_overall_r2 = r2
                best_overall_fold_error = fold_error
                best_overall_test_size = test_size
                best_overall_random_state = random_state
                best_overall_k = k
                best_features = selected_features.copy()

    print("\nBest Model Performance:")
    print(f"Best R-squared (R²): {best_overall_r2:.4f}")
    print(f"Best Test Size: {best_overall_test_size}")
    print(f"Best Random State: {best_overall_random_state}")
    print(f"Number of Features (k): {best_overall_k}")
    print(f"Selected Features: {list(best_features)}")
    print(f"Best Average Absolute Fold Error: {best_overall_fold_error:.4f}")
