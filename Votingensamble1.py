import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# --- 1. INSTALACIÓN Y CARGA ---
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

# --- 2. PREPROCESAMIENTO ---
print("\n--- 2. PREPROCESAMIENTO ---")
# Convertir a numérico
for col in df_full_train.columns:
    df_full_train[col] = pd.to_numeric(df_full_train[col], errors='coerce')
    if col in df_test.columns:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

# Rellenar NaN con MEDIANA
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

# Escalar (Necesario para LR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# --- 3. DEFINICIÓN DEL "TRIDENTE" ---
print("\n--- 3. DEFINICIÓN DEL TRIDENTE (LR + RF + XGB) ---")

# Modelo 1: Logistic Regression (Base Lineal)
clf_lr = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=3000,
    C=1.0,
    random_state=42
)

# Modelo 2: Random Forest (Varianza / No Linealidad)
clf_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Modelo 3: XGBoost (Bias / Boosting)
# Ajustado para ser conservador y evitar overfitting
clf_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,            # Poca profundidad
    learning_rate=0.05,     # Aprendizaje lento
    objective='multi:softprob',
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Voting Classifier
eclf = VotingClassifier(
    estimators=[
        ('lr', clf_lr), 
        ('rf', clf_rf), 
        ('xgb', clf_xgb)
    ],
    voting='soft'
)

# --- 4. ENTRENAMIENTO FINAL ---
print("\n--- 4. ENTRENAMIENTO FINAL (FULL DATA) ---")
eclf.fit(X_scaled, y)
print("Entrenamiento completado con el Tridente.")

# --- 5. SUBMISSION ---
print("\n--- 5. GENERACIÓN DEL SUBMISSION ---")
final_preds = eclf.predict(X_test_scaled)

# Crear DataFrame de salida con ID en mayúsculas
submission = pd.DataFrame({
    'ID': df_test.index,
    'label': final_preds
})

# Guardar
submission.to_csv('submission_trident_xgboost.csv', index=False)
print("Saved 'submission_trident_xgboost.csv'")

print("\nFirst 5 rows:")
print(submission.head())
