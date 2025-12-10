import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. CARGA Y FUSIÓN SIMPLE ---
print("--- 1. CARGA Y FUSIÓN ---")
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    
    try:
        df_external = pd.read_csv('external_data.csv')
        print("External data loaded.")
        df_train_total = pd.concat([df_train, df_external], axis=0, ignore_index=True)
    except FileNotFoundError:
        print("External data not found. Using only train.csv.")
        df_train_total = df_train.copy()

    # Eliminar columnas de ID si existen (patrones comunes)
    for col in df_train_total.columns:
        if 'id' in col.lower() or 'patient' in col.lower():
            print(f"Dropping ID column: {col}")
            df_train_total.drop(columns=[col], inplace=True)
            if col in df_test.columns:
                df_test.drop(columns=[col], inplace=True)

    print(f"Train shape: {df_train_total.shape}")
    print(f"Test shape: {df_test.shape}")

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

# --- 2. LIMPIEZA "BRUTA" ---
print("\n--- 2. LIMPIEZA BRUTA ---")
# Convertir a numérico (errors='coerce')
for col in df_train_total.columns:
    df_train_total[col] = pd.to_numeric(df_train_total[col], errors='coerce')
    if col in df_test.columns: # Asegurar que test tenga las mismas columnas (menos target)
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

# Rellenar NaN con MEDIANA
medians = df_train_total.median()
df_train_total.fillna(medians, inplace=True)
df_test.fillna(medians, inplace=True)

print("NaNs in Train:", df_train_total.isnull().sum().sum())
print("NaNs in Test:", df_test.isnull().sum().sum())

# --- 3. PREPARACIÓN (Estandarización) ---
print("\n--- 3. PREPARACIÓN ---")
target_col = 'label' # Asumimos 'label' por la descripción del usuario
if target_col not in df_train_total.columns:
    # Intentar buscar la columna target si tiene otro nombre, pero por defecto 'label'
    print("Target column 'label' not found!")
    exit()

X = df_train_total.drop(columns=[target_col])
y = df_train_total[target_col]

# Alinear columnas de test con train
X_test_final = df_test[X.columns]

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# --- 4. MODELADO (Maximizando Accuracy) ---
print("\n--- 4. MODELADO ---")
# Split de validación
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression 
model = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=3000,
    random_state=42
    # NO class_weight='balanced'
)

model.fit(X_train, y_train)

# --- 5. EVALUACIÓN Y PREDICCIÓN ---
print("\n--- 5. EVALUACIÓN ---")
y_pred_val = model.predict(X_val)

acc = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {acc:.5f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred_val)
print(cm)

with open('kaggle_results.txt', 'w') as f:
    f.write(f"Validation Accuracy: {acc:.5f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
print("Results saved to 'kaggle_results.txt'")

# Predicción Final
print("\n--- PREDICCIÓN FINAL ---")
final_preds = model.predict(X_test_scaled)

df_sub = pd.DataFrame({'prediction': final_preds})
df_sub.to_csv('submission_kaggle_logistic.csv', index=False)
print("Saved 'submission_kaggle_logistic.csv'")
