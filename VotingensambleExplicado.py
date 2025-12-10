import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# =============================================================================
# ARQUITECTURA DEL MODELO FINAL (ENSEMBLE)
# =============================================================================
# Este script muestra exclusivamente cómo se construye y valida el modelo
# que logró el 0.62 de Accuracy, sin la parte de generar el archivo de Kaggle.
# =============================================================================

print("--- 1. PREPARACIÓN DE DATOS (TRAIN + EXTERNAL) ---")
# Cargamos los datos de entrenamiento originales y los externos
df_train = pd.read_csv('train.csv')
df_statlog = pd.read_csv('statlog_limpio.csv')

# FUSIÓN: Unimos ambos datasets para tener más ejemplos de entrenamiento.
# Esto es clave: 'statlog' aporta muchos casos de clases 0 y 1.
df_full_train = pd.concat([df_train, df_statlog], axis=0, ignore_index=True)

# LIMPIEZA BÁSICA
# 1. Eliminar IDs (no sirven para predecir)
for col in df_full_train.columns:
    if 'id' in col.lower() or 'patient' in col.lower():
        df_full_train.drop(columns=[col], inplace=True)

# 2. Convertir todo a numérico (manejo de errores de tipeo)
for col in df_full_train.columns:
    df_full_train[col] = pd.to_numeric(df_full_train[col], errors='coerce')

# 3. Imputación de Nulos (NaN) con la MEDIANA
# Usamos la mediana porque es resistente a valores extremos (outliers).
medians = df_full_train.median()
df_full_train.fillna(medians, inplace=True)

# SEPARACIÓN X (Features) e y (Target)
target_col = 'label'
X = df_full_train.drop(columns=[target_col])
y = df_full_train[target_col]

# ESCALADO (StandardScaler)
# Fundamental para que la Regresión Logística funcione bien.
# Transforma los datos para que tengan media 0 y desviación estándar 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Datos procesados: {X_scaled.shape[0]} muestras, {X_scaled.shape[1]} características.")


print("\n--- 2. DEFINICIÓN DEL ENSEMBLE (EL CEREBRO) ---")

# COMPONENTE 1: Regresión Logística (El Experto Lineal)
# - Bueno para separar tendencias generales y clases bien definidas (0 y 1).
clf_lr = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial', # Maneja las 5 clases (0-4) nativamente
    max_iter=3000,             # Damos tiempo para que converja
    C=1.0,                     # Regularización estándar
    random_state=42
)

# COMPONENTE 2: Random Forest (El Experto No Lineal)
# - Bueno para encontrar patrones complejos y excepciones que la línea recta no ve.
clf_rf = RandomForestClassifier(
    n_estimators=200, # 200 árboles votando
    max_depth=10,     # Limitamos profundidad para no memorizar (overfitting)
    random_state=42
)

# VOTING CLASSIFIER (El Jefe)
# Combina los dos modelos anteriores.
# voting='soft': NO cuenta votos (1 vs 1), sino que PROMEDIA las confianzas.
# Ejemplo: Si LR dice "Clase 0 con 90% seguridad" y RF dice "Clase 0 con 60%",
# el promedio es alto (75%) y gana la Clase 0.
model_ensemble = VotingClassifier(
    estimators=[
        ('logistic_regression', clf_lr),
        ('random_forest', clf_rf)
    ],
    voting='soft'
)


print("\n--- 3. VALIDACIÓN DEL MODELO (PRUEBA DE RENDIMIENTO) ---")
# Usamos Cross-Validation (5 pliegues) para ver qué tan bueno es el modelo realmente.
# Esto divide los datos en 5 partes, entrena en 4 y prueba en 1, rotando 5 veces.
cv_scores = cross_val_score(model_ensemble, X_scaled, y, cv=5, scoring='accuracy')

print(f"Resultados de Validación Cruzada (5-Fold):")
print(f"Scores individuales: {cv_scores}")
print(f"ACCURACY PROMEDIO: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\nEste es el código puro del modelo. Si ejecutas esto, verás la métrica de 0.62 sin generar archivos de submission.")
