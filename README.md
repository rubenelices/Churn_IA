# Churn Predictor Pro

## [Acceder a la aplicación](https://churnia-aws6mxwfbck79eknhmujrq.streamlit.app/)

> **Link directo:** https://churnia-aws6mxwfbck79eknhmujrq.streamlit.app/

---

## Descripción

Aplicación de predicción de churn (abandono de clientes) desarrollada con Machine Learning para el sector automovilístico. Permite identificar qué clientes tienen mayor probabilidad de abandonar la empresa y diseñar estrategias comerciales para retenerlos.

## Secciones de la aplicación

- **Resumen del Proyecto** — Visión general, contexto del problema y arquitectura de datos (DataLake → Datawarehouse → Datamart)
- **Dashboard** — Métricas y comparativa de los modelos entrenados (AUC, F1, Precision, Recall)
- **Simulador de Riesgo** — Predicción individual de churn introduciendo los datos de un cliente manualmente
- **Análisis Nuevos Clientes** — Carga masiva de nuevos clientes para predecir su probabilidad de churn
- **Acción Comercial** — Estrategia de retención con segmentación de clientes, priorización por ROI y ranking de clientes a contactar

## Estructura del proyecto

```
├── app.py                          # Aplicación principal Streamlit
├── requirements.txt                # Dependencias
├── Prediccion_churn.ipynb          # Notebook con el desarrollo del modelo
├── Accion_Comercial.ipynb          #Notebook con la estrategia comercial y el calculo del ROI
├── Data/
│   ├── DataLake/                   # Datos originales en crudo
│   ├── Datawarehouse/              # Datos procesados y dimensiones
│   └── Datamart/                   # Datamart final para modelado
└── Graficas/                       # Visualizaciones exportadas
```

## Modelos utilizados

- XGBoost
- LightGBM
- Random Forest

## Tecnologías

- Python 3
- Streamlit
- Pandas / NumPy
- Scikit-learn
- XGBoost / LightGBM
- Plotly / Matplotlib
