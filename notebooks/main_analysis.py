"""
main_analysis.py
================
Predictive Analytics for Household Appliance Energy Consumption Using IoT Sensor Data.

This script:
1. Loads the energy consumption dataset (energydata_complete.csv)
2. Preprocesses the data (drops irrelevant columns, handles train-test split)
3. Scales features using StandardScaler
4. Trains a Random Forest Regressor model
5. Evaluates the model using MAE, RMSE, and R² score
6. Saves the trained model and scaler as .pkl (pickle) files for deployment
7. Verifies the saved .pkl files by loading and running sample inference
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_dataset():
    """Find the dataset path whether executed from notebooks/ or root directory."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "Datasets", "energydata_complete.csv"),
        os.path.join(os.path.dirname(__file__), "..", "energydata_complete.csv"),
        os.path.join("Datasets", "energydata_complete.csv"),
        "energydata_complete.csv",
        os.path.join(os.path.dirname(__file__), "energydata_complete.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find 'energydata_complete.csv'. Please ensure the dataset file is present."
    )


def main():
    print("=" * 60)
    print(" Appliance Energy Consumption Prediction — Model Training")
    print("=" * 60)

    # 1. Load Data
    data_path = find_dataset()
    print(f"\n[1/6] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2. Preprocess Data
    print("\n[2/6] Preprocessing data...")
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    # Drop columns not used in modeling:
    # - date: datetime object
    # - rv1, rv2: random noise variables
    # - hour: if present
    cols_to_drop = [col for col in ['date', 'rv1', 'rv2', 'hour'] if col in df.columns]
    df_model = df.drop(columns=cols_to_drop)

    X = df_model.drop(columns=['Appliances'])
    y = df_model['Appliances']
    feature_names = list(X.columns)
    print(f"      Target: Appliances")
    print(f"      Number of Features: {len(feature_names)}")
    print(f"      Feature Names: {feature_names}")

    # 3. Train-Test Split (80% Train, 20% Test)
    print("\n[3/6] Splitting dataset (80% train, 20% test, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train samples: {X_train.shape[0]}")
    print(f"      Test samples:  {X_test.shape[0]}")

    # 4. Feature Scaling
    print("\n[4/6] Fitting StandardScaler on training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Random Forest Model
    print("\n[5/6] Training Random Forest Regressor (n_estimators=100, max_depth=15)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    print("      Model training complete.")

    # Model Evaluation
    y_pred = rf_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print("        MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Mean Absolute Error (MAE) : {mae:.2f} Wh")
    print(f"  Mean Squared Error (MSE)  : {mse:.2f} Wh²")
    print(f"  Root Mean Squared (RMSE)  : {rmse:.2f} Wh")
    print(f"  R² Score                  : {r2:.4f} ({r2*100:.2f}% variance explained)")
    print("=" * 40)

    # 6. Save Model & Scaler to .pkl in models/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    os.makedirs(output_dir, exist_ok=True)
    
    model_pkl_path = os.path.join(output_dir, "appliance_energy_predictor.pkl")
    scaler_pkl_path = os.path.join(output_dir, "appliance_energy_scaler.pkl")
    bundle_pkl_path = os.path.join(output_dir, "appliance_energy_pipeline.pkl")

    print("\n[6/6] Saving trained model and scaler as .pkl files...")
    
    # Save appliance_energy_predictor.pkl
    with open(model_pkl_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"      [OK] Saved model to:  {model_pkl_path}")

    # Save appliance_energy_scaler.pkl
    with open(scaler_pkl_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"      [OK] Saved scaler to: {scaler_pkl_path}")

    # Save bundle (model + scaler + features + metrics)
    bundle = {
        "model": rf_model,
        "scaler": scaler,
        "features": feature_names,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2}
    }
    with open(bundle_pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"      [OK] Saved bundle to: {bundle_pkl_path}")

    # 7. Verification: Load and test inference
    print("\n--- Verifying Saved .pkl Files ---")
    with open(model_pkl_path, "rb") as f:
        loaded_model = pickle.load(f)
    with open(scaler_pkl_path, "rb") as f:
        loaded_scaler = pickle.load(f)

    sample_features = X_test.iloc[:1]
    sample_scaled = loaded_scaler.transform(sample_features)
    sample_pred = loaded_model.predict(sample_scaled)[0]
    actual_val = y_test.iloc[0]

    print(f"  Test Sample Actual Value    : {actual_val:.1f} Wh")
    print(f"  Test Sample Predicted Value : {sample_pred:.1f} Wh")
    print("  Verification successful! Model ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
