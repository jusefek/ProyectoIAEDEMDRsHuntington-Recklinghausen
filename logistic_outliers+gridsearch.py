import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. PREPARACIÓN DE DATOS ---
print("--- 1. PREPARACIÓN DE DATOS ---")
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

    # Eliminar IDs
    for col in df_train_total.columns:
        if 'id' in col.lower() or 'patient' in col.lower():
            df_train_total.drop(columns=[col], inplace=True)
            if col in df_test.columns:
                df_test.drop(columns=[col], inplace=True)

    # Convertir a numérico
    for col in df_train_total.columns:
        df_train_total[col] = pd.to_numeric(df_train_total[col], errors='coerce')
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    # Rellenar NaN con MEDIANA
    medians = df_train_total.median()
    df_train_total.fillna(medians, inplace=True)
    df_test.fillna(medians, inplace=True)

    # Eliminar duplicados
    initial_len = len(df_train_total)
    df_train_total.drop_duplicates(inplace=True)
    print(f"Dropped {initial_len - len(df_train_total)} duplicate rows.")

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

# --- 2. LIMPIEZA DE OUTLIERS (NUEVO) ---
print("\n--- 2. LIMPIEZA DE OUTLIERS ---")
# Filtros: chol > 500 o trestbps > 200
# Asegurarse de que las columnas existen
mask_outliers = pd.Series([False] * len(df_train_total), index=df_train_total.index)

if 'chol' in df_train_total.columns:
    mask_outliers |= (df_train_total['chol'] > 500)
if 'trestbps' in df_train_total.columns:
    mask_outliers |= (df_train_total['trestbps'] > 200)

n_outliers = mask_outliers.sum()
if n_outliers > 0:
    print(f"Removing {n_outliers} outliers (chol > 500 or trestbps > 200).")
    df_train_total = df_train_total[~mask_outliers]
else:
    print("No outliers found with specified criteria.")

print(f"Final Train shape: {df_train_total.shape}")

# --- 3. PRE-PROCESADO ---
print("\n--- 3. PRE-PROCESADO ---")
target_col = 'label'
if target_col not in df_train_total.columns:
    print("Target column not found!")
    exit()

X = df_train_total.drop(columns=[target_col])
y = df_train_total[target_col]

# Alinear test
X_test_final = df_test[X.columns]

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)

# --- 4. BÚSQUEDA DEL MEJOR MODELO (GRIDSEARCH) ---
print("\n--- 4. GRIDSEARCH CV ---")

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'newton-cg'],
    'multi_class': ['multinomial', 'ovr']
}

# Algunos solvers no soportan todas las combinaciones de multi_class/penalty, 
# pero GridSearchCV manejará los errores o warnings.
# liblinear no soporta multinomial, así que esperamos warnings o errores si no se filtra.
# Para evitar errores fatales, ajustaremos la grid o dejaremos que sklearn lo maneje (liblinear + multinomial lanza error).
# Mejor ajustamos la grid para ser seguros:
# Separar en dos grids si queremos ser muy precisos, o simplemente quitar liblinear si queremos multinomial real.
# El usuario pidió explícitamente esa lista. 
# Scikit-learn lanzará ValueError si la combinación es inválida.
# Vamos a ser proactivos y usar una lista de diccionarios para combinaciones válidas.

grid_params = [
    {'solver': ['lbfgs', 'newton-cg'], 'multi_class': ['multinomial', 'ovr'], 'C': [0.01, 0.1, 1, 10, 100]},
    {'solver': ['liblinear'], 'multi_class': ['ovr'], 'C': [0.01, 0.1, 1, 10, 100]} # liblinear only supports ovr
]

model = LogisticRegression(max_iter=5000, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=grid_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_scaled, y)

# --- 5. ENTRENAMIENTO Y RESULTADOS ---
print("\n--- 5. RESULTADOS ---")
print(f"¿Cuál fue la mejor combinación de parámetros? {grid_search.best_params_}")
print(f"¿Cuál es el mejor Accuracy obtenido en validación? {grid_search.best_score_:.5f}")

# Guardar resultados en txt
with open('gridsearch_results.txt', 'w') as f:
    f.write(f"Best Params: {grid_search.best_params_}\n")
    f.write(f"Best CV Accuracy: {grid_search.best_score_:.5f}\n")

# Predicción con el mejor modelo
best_model = grid_search.best_estimator_
final_preds = best_model.predict(X_test_scaled)

df_sub = pd.DataFrame({'prediction': final_preds})
df_sub.to_csv('submission_gridsearch.csv', index=False)
print("Saved 'submission_gridsearch.csv'")
