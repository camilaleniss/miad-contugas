# MIAD Contugas: Detección de Anomalías en Consumo de Gas

Este es un proyecto de machine learning para detectar anomalías en el consumo de gas de clientes comerciales e industriales.

## Resumen del Proyecto

El objetivo principal es identificar patrones de consumo de gas anómalos utilizando modelos de machine learning. El proyecto está estructurado en varias fases: procesamiento de datos (ETL), análisis exploratorio, entrenamiento de modelos y visualización de resultados.

## Datos

*   **Fuente:** El archivo `df_contugas.csv` contiene los datos de consumo con 847,946 registros.
*   **Características:** Los datos son horarios y contienen las siguientes columnas: `Fecha`, `Cliente`, `Segmento`, `Presion`, `Temperatura` y `Volumen`.
*   **Análisis Exploratorio:** El documento `Supuestos_Modelo_Gas_compat.pdf` indica que:
    *   La variable `Volumen` (consumo) no sigue una distribución normal, tiene muchos ceros y valores atípicos.
    *   La `Presion` tiene una distribución bimodal.
    *   La `Temperatura` es aproximadamente gaussiana.
    *   Existe una correlación moderada entre el `Volumen` y la `Temperatura`, y una correlación negativa con la `Presion`.

## Metodología

*   **Modelado por Cliente:** El análisis se realiza de forma individual para cada cliente.
*   **Modelos Utilizados:**
    *   Se emplea un modelo multivariable con `Temperatura` y `Presion` como variables exógenas.
    *   Se utiliza **Regresión Robusta** para manejar los outliers y la no normalidad de los datos.
    *   Se aplica **Isolation Forest**, un algoritmo no supervisado, para la detección de anomalías.
    *   Se contempla el uso de **LSTM** para capturar dependencias temporales más complejas.
*   **Detección de Anomalías:** Se utiliza un umbral robusto basado en la Desviación Absoluta Mediana (MAD) sobre los residuos del modelo para marcar una observación como anomalía.
*   **Validación:** Se realiza *backtesting* inyectando anomalías sintéticas para evaluar la precisión y el *recall* del modelo.

## Estructura de Carpetas

*   `etl/`: Contiene el script `etlcontugas.py` para el procesamiento de los datos.
*   `supuestos/`: Incluye un notebook de Jupyter (`EDA_Gas_Multivariado.ipynb`) para el análisis exploratorio y un documento con los supuestos del modelo.
*   `Training/`: Contiene el pipeline de entrenamiento (`run_pipeline.py`), un script de evaluación con datos sintéticos (`main_eval_injection.py`) y los modelos guardados.
*   `dashboard/`: Alberga una aplicación de Streamlit (`streamlit_app.py`) para la visualización de los resultados.

## Instrucciones de Ejecución

1.  **Entrenar el modelo:**
    ```bash
    python run_pipeline.py config.yaml
    ```
2.  **Evaluar el modelo:**
    ```bash
    python main_eval_injection.py ./df_anom_injection.parquet
    ```
3.  **Visualizar experimentos con MLflow:**
    ```bash
    python -m mlflow ui --backend-store-uri file:./mlruns --port 5000
    ```
4.  **Ejecutar el dashboard:**
    ```bash
    streamlit run src/streamlit_app.py
    ```