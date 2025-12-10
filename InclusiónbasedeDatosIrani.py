import pandas as pd
import numpy as np

# Cargar el dataset original
input_file = 'Z-Alizadeh sani dataset.xlsx'
sheet_name = 'Sheet 1 - Table 1'
print(f"Cargando {input_file} ({sheet_name})...")
try:
    df_orig = pd.read_excel(input_file, sheet_name=sheet_name)
except FileNotFoundError:
    print(f"Error: No se encuentra el archivo {input_file}")
    exit(1)
except Exception as e:
    print(f"Error cargando el archivo: {e}")
    exit(1)

print(f"Filas originales: {len(df_orig)}")

# Crear DataFrame destino
df_new = pd.DataFrame()

# --- Mapeo de columnas ---

# age: Copia la columna 'Age'
df_new['age'] = df_orig['Age']

# sex: Transforma la columna 'Sex': "Male" a 1, "Female" a 0
df_new['sex'] = df_orig['Sex'].map({'Male': 1, 'Female': 0})

# cp: Derivar de 'Typical Chest Pain', 'Atypical', 'Nonanginal'
# Typical Angina -> 1
# Atypical -> 2
# Non-anginal -> 3
# Asymptomatic -> 4 (Si ninguno de los anteriores es 'Y')

def get_cp(row):
    if str(row.get('Typical Chest Pain', '')).lower() == 'y':
        return 1
    elif str(row.get('Atypical', '')).lower() == 'y':
        return 2
    elif str(row.get('Nonanginal', '')).lower() == 'y':
        return 3
    else:
        return 4

df_new['cp'] = df_orig.apply(get_cp, axis=1)

# trestbps: Usa la columna 'BP'
df_new['trestbps'] = df_orig['BP']

# chol: Usa la columna 'TG' (triglicéridos)
df_new['chol'] = df_orig['TG']

# fbs: Usa la columna 'DM'. Si es "Yes" o "1", pon 1; si no, 0.
df_new['fbs'] = df_orig['DM'].astype(str).apply(lambda x: 1 if x.lower() in ['yes', '1'] else 0)

# thalach: Usa la columna 'PR'
df_new['thalach'] = df_orig['PR']

# exang: Usa la columna 'Exertional CP'. Si es "Y" o "Yes", pon 1; si no, 0.
df_new['exang'] = df_orig['Exertional CP'].astype(str).apply(lambda x: 1 if x.lower() in ['y', 'yes'] else 0)

# restecg, oldpeak, slope, ca, thal: Estas columnas no existen en el origen. Créalas y llénalas con valores vacíos (NaN).
df_new['restecg'] = np.nan
df_new['oldpeak'] = np.nan
df_new['slope'] = np.nan
df_new['ca'] = np.nan
df_new['thal'] = np.nan

# --- Lógica para la columna 'label' (Severidad) ---
# Usa la columna 'Cath'.
# Si 'Cath' es igual a 'Cad', asigna label = 1 (Enfermo).
# Si 'Cath' es igual a 'Normal', asigna label = 0 (Sano).

df_new['label'] = df_orig['Cath'].apply(lambda x: 1 if str(x).lower() == 'cad' else 0)

# --- Filtrado final ---
# Filtra y guarda en el CSV final SOLO las filas donde label == 1.
print(f"Filas antes de filtrar sanos: {len(df_new)}")
df_sick = df_new[df_new['label'] == 1].copy()
print(f"Filas después de filtrar (label == 1): {len(df_sick)}")

# Ordenar columnas según especificación
cols_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
df_sick = df_sick[cols_order]

# Guardar
output_file = 'statlog_extra_sick.csv'
df_sick.to_csv(output_file, index=False)
print(f"Archivo guardado: {output_file}")

# Verificación rápida
print("\nPrimeras 5 filas:")
print(df_sick.head())
print("\nConteo de labels:")
print(df_sick['label'].value_counts())
print("\nConteo de CP:")
print(df_sick['cp'].value_counts())
