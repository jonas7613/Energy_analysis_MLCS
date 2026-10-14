# Household Appliance Energy Consumption Prediction

Predictive machine learning model to estimate household appliance energy consumption (in Watt-hours) using IoT temperature and humidity sensors across 9 rooms and outdoor weather data.

---

## Repository Structure

```text
Energy_Analysis_MLCS/
├── Datasets/
│   └── energydata_complete.csv          # 10-minute interval sensor readings (~4.5 months)
├── models/
│   ├── appliance_energy_predictor.pkl   # Trained Random Forest Regressor
│   ├── appliance_energy_scaler.pkl      # Fitted StandardScaler for input features
│   └── appliance_energy_pipeline.pkl    # Bundle containing model, scaler & feature list
├── notebooks/
│   ├── appliance_energy_analysis.ipynb  # Exploratory Data Analysis & visual evaluation
│   └── main_analysis.py                 # Training script & model exporter
└── README.md
```

---

## Model & Performance

- **Algorithm**: Random Forest Regressor (`n_estimators=100`, `max_depth=15`, `random_state=42`)
- **Preprocessing**: Datetime parsing, irrelevant columns removed (`date`, `rv1`, `rv2`), feature scaling with `StandardScaler`.
- **Evaluation Metrics**:
  - **Mean Absolute Error (MAE)**: ~34.99 Wh
  - **Root Mean Squared Error (RMSE)**: ~71.06 Wh
  - **$R^2$ Score**: ~0.495 (~49.5% variance explained)

---

## Quickstart

### 1. Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Train and Save Models
```bash
python notebooks/main_analysis.py
```

### 3. Load & Run Inference
```python
import pickle

# Load model and scaler
with open("models/appliance_energy_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/appliance_energy_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Predict on new sensor values (25 features)
scaled_input = scaler.transform(sample_features)
predicted_energy = model.predict(scaled_input)
print(f"Predicted Consumption: {predicted_energy[0]:.2f} Wh")
```
