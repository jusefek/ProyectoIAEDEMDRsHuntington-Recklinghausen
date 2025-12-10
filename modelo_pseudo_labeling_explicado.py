import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# =============================================================================
# ESTRATEGIA AVANZADA: SEMI-SUPERVISED LEARNING (PSEUDO-LABELING)
# =============================================================================
# OBJETIVO: Aprovechar los datos del TEST (sin etiqueta) para mejorar el entrenamiento.
#
# ¿CÓMO FUNCIONA?
# 1. Entrenamos un modelo "profesor" con los datos que tenemos.
# 2. El profesor hace el examen (predice sobre test).
# 3. Seleccionamos las respuestas donde el profesor está MUY SEGURO (>90%).
# 4. Asumimos que esas respuestas son correctas (Pseudo-Labels) y las añadimos al entrenamiento.
# 5. Re-entrenamos al modelo (ahora "alumno") con más datos.
# =============================================================================

print("--- 1. PREPARACIÓN DE DATOS ---")
# Carga de datos
df_train = pd.read_csv('train.csv')
df_statlog = pd.read_csv('statlog_limpio.csv') # Datos externos
df_extra_sick = pd.read_csv('statlog_extra_sick.csv') # Datos extra enfermos (Z-Alizadeh)
df_test = pd.read_csv('test.csv') # Datos sin etiqueta (el examen)

# --- CORRECCIÓN DE SESGO ---
# Eliminar filas de extra_sick donde cp == 4 (Asintomático)
print(f"Extra sick original: {len(df_extra_sick)}")
df_extra_sick = df_extra_sick[df_extra_sick['cp'] != 4]
print(f"Extra sick filtrado (sin CP=4): {len(df_extra_sick)}")

# FUSIÓN: Unimos train, statlog y extra_sick filtrado
df_train_full = pd.concat([df_train, df_statlog, df_extra_sick], axis=0, ignore_index=True)

# --- PRESERVAR IDs ---
# Buscamos la columna ID en el test antes de limpiar
test_ids = None
for col in df_test.columns:
    if 'id' in col.lower() or 'patient' in col.lower():
        test_ids = df_test[col].copy()
        print(f"IDs encontrados en columna: {col}")
        break

if test_ids is None:
    print("No se encontró columna ID en test.csv. Generando IDs secuenciales.")
    test_ids = range(len(df_test))

# LIMPIEZA
# - Eliminar IDs y 'chol'
# - Convertir a numérico
# - Rellenar con MEDIANA

# Eliminar 'chol' si existe
if 'chol' in df_train_full.columns:
    df_train_full.drop(columns=['chol'], inplace=True)
if 'chol' in df_test.columns:
    df_test.drop(columns=['chol'], inplace=True)

# Eliminar columnas de ID del dataset de trabajo
for col in df_train_full.columns:
    if 'id' in col.lower() or 'patient' in col.lower():
        df_train_full.drop(columns=[col], inplace=True)
        if col in df_test.columns:
            df_test.drop(columns=[col], inplace=True)
    
    # Asegurar numérico
    df_train_full[col] = pd.to_numeric(df_train_full[col], errors='coerce')
    if col in df_test.columns:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

medians = df_train_full.median()
df_train_full.fillna(medians, inplace=True)
df_test.fillna(medians, inplace=True)

# SEPARACIÓN X e y
target_col = 'label'
X = df_train_full.drop(columns=[target_col])
y = df_train_full[target_col]

# Alinear test
X_test_final = df_test[X.columns]

# ESCALADO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_final)


print("\n--- 2. DEFINICIÓN DEL MODELO BASE (EL PROFESOR) ---")
# Usamos nuestro mejor Ensemble: Regresión Logística + Random Forest
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=5000, C=1.0, random_state=42)
clf2 = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)

model = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2)],
    voting='soft' # Importante: 'soft' para obtener probabilidades
)


print("\n--- 3. FASE 1: ENTRENAMIENTO INICIAL ---")
model.fit(X_scaled, y)
print("El profesor ha estudiado los datos etiquetados.")


print("\n--- 4. FASE 2: PSEUDO-LABELING (EL TRUCO) ---")
# El modelo predice las PROBABILIDADES sobre el test
probs = model.predict_proba(X_test_scaled)
preds = model.predict(X_test_scaled)

# Buscamos muestras con confianza ALTA
threshold = 0.90 # 90% de seguridad
high_conf_indices = np.where(np.max(probs, axis=1) > threshold)[0]

# Si hay pocas, bajamos un poco la vara para no quedarnos sin datos extra
if len(high_conf_indices) < 50:
    print(f"Pocas muestras seguras. Bajando exigencia al 85%...")
    threshold = 0.85
    high_conf_indices = np.where(np.max(probs, axis=1) > threshold)[0]

print(f"¡Encontradas {len(high_conf_indices)} muestras en el Test con confianza > {threshold*100}%!")

# Extraemos esas muestras y sus predicciones (que ahora son sus 'etiquetas')
X_pseudo_scaled = X_test_scaled[high_conf_indices]
y_pseudo = preds[high_conf_indices]

# AUMENTO DE DATOS
# Unimos los datos originales (X_scaled) con los nuevos datos pseudo-etiquetados (X_pseudo_scaled)
X_augmented = np.vstack((X_scaled, X_pseudo_scaled))
y_augmented = np.concatenate((y, y_pseudo))

print(f"Dataset original: {X_scaled.shape[0]} muestras.")
print(f"Dataset aumentado: {X_augmented.shape[0]} muestras.")


print("\n--- 5. FASE 3: RE-ENTRENAMIENTO FINAL ---")
# El modelo vuelve a estudiar, ahora con más material (incluyendo lo que "aprendió" del test)
model.fit(X_augmented, y_augmented)
print("El modelo ha sido re-entrenado con éxito (Semi-Supervised Learning).")

# --- 6. GENERACIÓN DE SUBMISSION ---
print("\n--- 6. GENERACIÓN DE SUBMISSION ---")
# Predecir sobre el test set original (escalado) usando el modelo re-entrenado
final_preds = model.predict(X_test_scaled)

# Crear DataFrame de submission usando los IDs preservados
submission = pd.DataFrame({
    'ID': test_ids,
    'label': final_preds
})

# Guardar
output_file = 'submissionUltimo.csv'
submission.to_csv(output_file, index=False)
print(f"Archivo de submission guardado en: {output_file}")
