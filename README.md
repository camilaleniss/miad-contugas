# MIAD Contugas: Detección de Anomalías en Consumo de Gas

Sistema de detección de anomalías en el consumo de gas comercial e industrial para la empresa Contugas (Perú). Este proyecto combina técnicas de análisis de series temporales (SSA), modelos de regresión Elastic Net para pronóstico y algoritmos de detección no supervisada (Isolation Forest) para identificar comportamientos anómalos en el consumo y variables operativas.

## 🎯 Características Principales

- **Detección de anomalías** por severidad (Leve, Media, Crítica) con umbrales configurables
- **Pronóstico vs. Real** (ENet) con análisis de residuales y bandas de confianza
- **Visualización interactiva** de datos históricos y anomalías detectadas
- **Análisis por cliente y segmento** (Comercial/Industrial)
- **Carga de nuevos datos** (ETL) para mantener el sistema actualizado
- **Exportación de reportes** en formato CSV
- **Dashboard web interactivo** desarrollado con Streamlit

## 📊 Datos

- **Fuente**: Archivo `df_contugas.csv` con 847,946 registros horarios
- **Variables**: Fecha, Cliente, Segmento, Presión, Temperatura, Volumen
- **Características**:
  - Volumen: No sigue distribución normal, presenta valores atípicos
  - Presión: Distribución bimodal
  - Temperatura: Aproximadamente gaussiana
  - Correlaciones: Volumen-Temperatura (positiva), Volumen-Presión (negativa)

## 🧠 Metodología

### Modelado por Cliente

- Análisis individual para cada cliente
- Modelo multivariable con Temperatura y Presión como variables exógenas
- Regresión Robusta para manejar outliers y no normalidad

### Detección de Anomalías

- **SSA (Singular Spectrum Analysis)**: Filtrado de series temporales
- **Elastic Net**: Modelo de pronóstico
- **Isolation Forest**: Detección no supervisada de anomalías
- **Umbral robusto**: Basado en Desviación Absoluta Mediana (MAD) sobre residuos

### Validación

- Backtesting con anomalías sintéticas inyectadas
- Evaluación de precisión y recall del modelo

## 🏗️ Estructura del Proyecto

```bash
miad-contugas/
├── dashboard/                 # Aplicación web principal
│   ├── src/
│   │   ├── streamlit_app.py  # Dashboard interactivo
│   │   ├── etl.py           # Procesamiento de datos
│   │   └── media/           # Recursos (logo, etc.)
│   ├── data/                # Datos procesados
│   ├── model_outputs/       # Modelos entrenados
│   ├── config/             # Configuraciones
│   └── requirements.txt    # Dependencias
├── models-training/         # Entrenamiento de modelos
│   ├── run_pipeline.py     # Pipeline de entrenamiento
│   ├── main_eval_injection.py # Evaluación con datos sintéticos
│   └── model_outputs/      # Modelos y métricas
├── etl/                    # Procesamiento inicial de datos
│   ├── dataset_contugas.xlsx # Dataset original
│   └── samples/            # Archivos de muestra para demo
│       └── df_test_contugas_2024_CLIENTE1 (Adicion de datos).xlsx
└── supuestos/             # Análisis exploratorio y documentación
```

## 🔧 Requisitos del Sistema

### Hardware Mínimo

- **CPU**: Procesador de doble núcleo (2 cores) a 2.0 GHz o superior
- **RAM**: 4 GB
- **Almacenamiento**: 1 GB de espacio libre

### Software

- **Python**: 3.10 o 3.11
- **Navegador Web**: Chrome, Firefox o Edge (versiones modernas)

### Librerías Principales

- `streamlit`: Dashboard interactivo
- `pandas`: Manipulación de datos
- `scikit-learn`: Modelos de machine learning
- `numpy`: Operaciones numéricas
- `plotly`: Gráficos interactivos
- `pyarrow`: Manejo de datos Parquet
- `mlflow`: Gestión de experimentos
- `openpyxl`: Lectura de archivos Excel
- `pyyaml`: Configuraciones

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone https://github.com/camilaleniss/miad-contugas.git
cd miad-contugas
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
cd dashboard
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación

```bash
streamlit run src/streamlit_app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## 📖 Guía de Uso

### Flujo de Trabajo Principal

#### 1. Seleccionar Cliente

En la barra lateral:

- **Segmento**: Elige entre Comercial o Industrial
- **Cliente**: Selecciona el cliente específico
- **Rango de fechas**: Define el período de análisis

#### 2. Analizar Anomalías

Ve a la pestaña **📊 Anomalías**:

- Revisa el resumen de anomalías (Leve, Media, Crítica)
- Observa el gráfico de predicción vs. real
- Identifica puntos rojos (anomalías detectadas)
- Descarga el reporte CSV con el botón "⬇️ Descargar anomalías"

#### 3. Explorar Histórico

En la pestaña **📈 Histórico**:

- Visualiza métricas clave del cliente
- Explora gráficos de Volumen, Temperatura y Presión
- Revisa estadísticas descriptivas

#### 4. Vista General

En la pestaña **👥 Resumen de Clientes**:

- Top 10 clientes por consumo
- Distribución por segmento
- Estadísticas generales

### Funcionalidades Avanzadas

#### Cargar Nuevos Datos

1. En la barra lateral, usa "Subir nuevas mediciones"
2. Selecciona archivo CSV o Excel (puedes usar el archivo de muestra en `etl/samples/`)
3. Presiona "Subir archivo" para procesar e integrar

**Archivo de muestra disponible**: `etl/samples/df_test_contugas_2024_CLIENTE1 (Adicion de datos).xlsx`

- Contiene datos de prueba para demostrar la funcionalidad de carga
- Formato compatible con el sistema ETL
- Útil para pruebas y demos del sistema

#### Ajustar Sensibilidad

En "Detección de anomalías":

- Modifica umbrales de severidad (Leve, Media, Crítica)
- Ajusta tolerancia de bandas (±%)
- Configura parámetros avanzados

## 🔧 Comandos de Desarrollo

### Entrenar Modelos

```bash
cd models-training
python run_pipeline.py config.yaml
```

### Evaluar Modelos

```bash
python main_eval_injection.py ./df_anom_injection.parquet
```

### Visualizar Experimentos (MLflow)

```bash
python -m mlflow ui --backend-store-uri file:./mlruns --port 5000
```

## 📋 Casos de Uso

### Supervisión Operativa

- Monitoreo continuo del consumo de gas
- Detección temprana de desviaciones
- Alertas por severidad de anomalías

### Análisis de Eficiencia

- Identificación de patrones de consumo anómalos
- Comparación entre clientes y segmentos
- Optimización de recursos operativos

### Reportes Gerenciales

- Exportación de datos de anomalías
- Métricas de rendimiento por cliente
- Análisis histórico y tendencias

## 🛠️ Mantenimiento

### Frecuencia Recomendada

- **Modelos**: Retrenamiento trimestral o cuando se detecten cambios significativos
- **Datos**: Actualización diaria/semanal según disponibilidad
- **Sistema**: Monitoreo continuo de rendimiento

### Roles Responsables

- **Analista de Datos**: Configuración y ajuste de parámetros
- **Operador**: Monitoreo diario y generación de reportes
- **Administrador**: Mantenimiento del sistema y actualizaciones

## 🐛 Resolución de Problemas

### Problemas Comunes

- **Error de memoria**: Aumentar RAM o procesar datos en lotes más pequeños
- **Modelos no encontrados**: Verificar que los archivos .pkl estén en `model_outputs/`
- **Datos no cargan**: Verificar formato CSV/Excel y codificación UTF-8

### Logs y Debugging

- Usa el expander "🔎 Auditoría de artefactos" en el dashboard
- Revisa la consola del navegador para errores JavaScript
- Verifica logs de Streamlit en la terminal

## 👥 Autores

- **Oscar Camilo Álvarez**
- **Ronaldo Ballesteros**
- **Maria Camila Lenis**
- **Julio César Solano**

**Universidad**: Universidad de los Andes - Departamento de Ingeniería Industrial

## 📞 Soporte

Para soporte técnico o consultas:

- Revisar este README y la documentación incluida
- Verificar requisitos del sistema
- Consultar logs de error para diagnóstico

## 📄 Licencia

Este proyecto es parte del Proyecto Aplicado en Analítica de Datos (PAAD) de la Universidad de los Andes.

---

**Nota importante**: Esta herramienta es un apoyo a la supervisión y toma de decisiones, no sustituye el análisis experto. Los resultados deben ser interpretados por personal calificado.
