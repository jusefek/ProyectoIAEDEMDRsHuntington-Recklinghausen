import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# --- 1. CARGA Y UNIÓN DE DATOS ---
print("--- 1. CARGA Y UNIÓN DE DATOS ---")
try:
    df_train = pd.read_csv('train.csv')
    df_statlog = pd.read_csv('statlog_limpio.csv')
    df_test = pd.read_csv('test.csv')
    
    # Concatena train y statlog
    df_full_train = pd.concat([df_train, df_statlog], axis=0, ignore_index=True)
    print(f"Full training set shape: {df_full_train.shape}")

    # Eliminar IDs si existen
    for col in df_full_train.columns:
        if 'id' in col.lower() or 'patient' in col.lower():
            df_full_train.drop(columns=[col], inplace=True)
            if col in df_test.columns:
                df_test.drop(columns=[col], inplace=True)

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

# --- 2. PREPROCESAMIENTO ROBUSTO ---
print("\n--- 2. PREPROCESAMIENTO ROBUSTO ---")
# Convertir a numérico
for col in df_full_train.columns:
    df_full_train[col] = pd.to_numeric(df_full_train[col], errors='coerce')
    if col in df_test.columns:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

# Rellenar NaN con MEDIANA (calculada sobre FULL TRAIN)
medians = df_full_train.median()
df_full_train.fillna(medians, inplace=True)
df_test.fillna(medians, inplace=True)

# Separar X e y
target_col = 'label'
if target_col not in df_full_train.columns:
    print("Target column not found!")
    exit()

X = df_full_train.drop(columns=[target_col])
y = df_full_train[target_col]

# Alinear test
X_test_final = df_test[X.columns]

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# --- 3. DEFINICIÓN DEL ENSEMBLE ---
print("\n--- 3. DEFINICIÓN DEL ENSEMBLE ---")
# Modelo 1: Logistic Regression
clf1 = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=3000,
    C=1.0,
    random_state=42
)

# Modelo 2: Random Forest
clf2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Voting Classifier
eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2)],
    voting='soft'
)

# --- 4. ENTRENAMIENTO FINAL ---
print("\n--- 4. ENTRENAMIENTO FINAL (FULL DATA) ---")
eclf.fit(X_scaled, y)
print("Model trained on 100% of data.")

# --- 5. GENERACIÓN DEL SUBMISSION ---
print("\n--- 5. GENERACIÓN DEL SUBMISSION ---")
final_preds = eclf.predict(X_test_scaled)

# Crear DataFrame de salida
# Usamos el índice de test como ID (0, 1, 2...)
submission = pd.DataFrame({
    'ID': df_test.index,
    'label': final_preds
})

# Guardar
submission.to_csv('submission_final_ensemble.csv', index=False)
print("Saved 'submission_final_ensemble.csv'")

print("\nFirst 5 rows:")
print(submission.head())
