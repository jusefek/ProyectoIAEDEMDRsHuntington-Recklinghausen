📂 Estructura del Proyecto: Detección de Enfermedades
🌟 El Archivo Principal
AAA_MODELO_FINAL.ipynb

🚨 Este es el entregable final. Este notebook consolida todo el trabajo exitoso. Integra la limpieza, el aumento de datos, y ejecuta el Voting Ensemble (nuestro mejor modelo) para generar las predicciones finales. Si solo se ejecuta un archivo, debe ser este.

🛠️ Desglose de Componentes
Para llegar al modelo final, dividimos el trabajo en módulos específicos:

1. Análisis y Preparación de Datos
EDA.ipynb: Análisis Exploratorio de Datos. Aquí diagnosticamos los problemas iniciales: dataset insuficiente (700 filas) y desbalanceado (5% casos graves).

limpieza_de_datos...py: Script encargado de estandarizar formatos y tratar valores nulos (MICE).

2. Estrategias de Mejora (Data Augmentation)
statlog+Ensamble.py & InclusiónbasedeDatosIrani.py:

Implementación de la solución al "Dataset Insuficiente".

Fusión con datasets externos (como STATLOG +270 pacientes) unificando los labels al formato del proyecto.

modelo_pseudo_labeling_explicado.py:

Técnica avanzada para aprovechar datos de test sin etiqueta (usando 90% de confianza) y reentrenar el modelo sin overfitting.

3. Modelos y Experimentos
Votingensamble1.py / VotingensambleExplicado.py:

La "joya de la corona". Combina la Regresión Logística y Random Forest/XGBoost para superar la barrera del 53% de acierto.

Logistic...py & Logistica+mediana.py:

Pruebas aisladas con modelos lineales y optimización de hiperparámetros (GridSearch).

🚀 Resumen del Flujo de Trabajo
Entrada: Datos crudos + Datasets Externos.

Proceso: Limpieza -> Data Augmentation -> Pseudo Labeling.

Modelo: Voting Ensemble (Logística + Random Forest).

Salida: AAA_MODELO_FINAL.ipynb ✅
