El presente documento detalla el enunciado y la metadata del caso práctico centrado en la predicción de fuga de clientes (Churn) en el sector de automoción, así como el diseño de una estrategia comercial rentable (Referencia: casos_practicos.pdf). El escenario plantea un concesionario oficial que busca mitigar la pérdida de clientes tanto en el área de postventa (taller) como en futuras operaciones de compra-venta de vehículos.

### Contexto del Caso Práctico
La organización dispone de un repositorio histórico estructurado que abarca información de ventas, perfiles sociodemográficos de los clientes, datos técnicos de los vehículos, registros de mantenimiento, gestión de incidencias o quejas, y un desglose detallado de márgenes y costes. El concepto central de este análisis es el Churn, que se define operativamente como una variable binaria donde un valor de 1 identifica a aquellos clientes que superan los 400 días sin registrar una revisión técnica en el taller oficial (Referencia: casos_practicos.pdf, pág. 1).

### Metodología y Objetivos del Alumno
El desarrollo del proyecto se divide en fases secuenciales que exigen al alumno la aplicación de técnicas avanzadas de ciencia de datos y visión estratégica de negocio. En la primera fase, se debe construir un modelo de Machine Learning capaz de predecir la probabilidad de fuga de cada cliente, justificando rigurosamente la selección de las variables predictoras y evaluando el desempeño mediante métricas de precisión, exhaustividad y curvas ROC. Posteriormente, el alumno determinará un umbral de decisión óptimo para clasificar a los clientes en distintos segmentos de riesgo.

En la segunda fase, el enfoque se desplaza hacia el diseño de la acción comercial. El alumno debe proponer una estrategia de actuación fundamentada en las probabilidades de Churn obtenidas, respetando estrictas condiciones económicas. El coste del mantenimiento preventivo se proyecta según la fórmula de capitalización compuesta $C(n) = BASE \cdot (1 + \alpha)^n$, donde el parámetro $\alpha$ varía entre el 7% para productos premium (A y B) y el 10% para el resto de la gama (Referencia: casos_practicos.pdf, pág. 2). 

$$C(n) = BASE \cdot (1 + \alpha)^n$$
```latex
C(n) = BASE \cdot (1 + \alpha)^n
```

La estrategia debe integrar costes de marketing, estimados inicialmente en un 1% del coste de mantenimiento, y considerar acciones adicionales como descuentos fijos de 1000€ para renovaciones de flota en clientes con alta fidelidad técnica ($n \ge 5$). Es imperativo que el margen neto del concesionario supere el 30%, tras destinar un 7% del ingreso bruto a la marca (Referencia: casos_practicos.pdf, pág. 2).

### Metadata de la Tabla Churn.csv
La tabla de datos integra la realidad operativa con los cálculos derivados de la estrategia de costes analizada. A continuación, se presenta el esquema detallado de las variables:

| Variable | Tipo de Dato | Nulos | Blancos | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| CODE | str | 0 | 0 | Identificador interno de la transacción comercial. |
| Sales_Date | str | 0 | 0 | Fecha original de la venta del vehículo. |
| Id_Producto | str | 0 | 0 | Código del artículo según la taxonomía del catálogo. |
| Customer_ID | int64 | 0 | 0 | Clave única de identificación del cliente. |
| PVP | int64 | 0 | 0 | Precio final de venta al público registrado. |
| MOTIVO_VENTA | str | 0 | 0 | Segmentación cualitativa de la venta (Particular/Empresa). |
| FORMA_PAGO | str | 0 | 0 | Método de abono (Contado, Financiera, etc.). |
| EXTENSION_GARANTIA | str | 0 | 0 | Estado de contratación de protección adicional (SI/NO). |
| SEGURO_BATERIA_LARGO_PLAZO | str | 0 | 0 | Cobertura técnica específica para vehículos electrificados. |
| MANTENIMIENTO_GRATUITO | int64 | 0 | 0 | Cantidad de servicios preventivos incluidos en la venta. |
| FIN_GARANTIA | str | 0 | 0 | Fecha de expiración de la cobertura oficial. |
| BASE_DATE | str | 0 | 0 | Fecha de corte para el análisis de antigüedad (31/12/2023). |
| EN_GARANTIA | str | 0 | 0 | Indicador binario de validez de garantía actual. |
| COSTE_VENTA_NO_IMPUESTOS | int64 | 0 | 0 | Gastos operativos directos de la transacción inicial. |
| GENERO | str | 849 | 0 | Rasgo demográfico para segmentación de campañas. |
| CODIGO_POSTAL | str | 0 | 0 | Localización geográfica para análisis de área de influencia. |
| Edad | int64 | 0 | 0 | Edad del titular en el momento de la consulta. |
| RENTA_MEDIA_ESTIMADA | int64 | 0 | 0 | Nivel de ingresos proyectado según entorno. |
| ENCUESTA_CLIENTE_ZONA_TALLER | int64 | 0 | 0 | Índice de satisfacción cualitativa percibida. |
| STATUS_SOCIAL | str | 12816 | 0 | Clasificación socioeconómica del perfil de cliente. |
| Modelo | str | 0 | 0 | Versión específica del vehículo (A-K). |
| TIPO_CARROCERIA | str | 0 | 0 | Formato estructural del producto comercializado. |
| Fuel | str | 0 | 0 | Fuente de propulsión (Combustión, Híbrido, Eléctrico). |
| Kw | int64 | 0 | 0 | Especificación técnica de potencia nominal. |
| Revisiones | int64 | 0 | 0 | Contador acumulado de entradas al taller oficial. |
| QUEJA | str | 33323 | 0 | Registro de insatisfacciones reportadas (SI/NO). |
| TIENDA_DESC | str | 0 | 0 | Denominación del punto de venta físico. |
| Margen_eur_bruto | float64 | 0 | 0 | Margen tras impuestos y comisiones de venta. |
| Margen_eur | float64 | 0 | 0 | Rentabilidad neta final retenida por el concesionario. |
| DAYS_LAST_SERVICE | float64 | 27070 | 0 | Días desde la última interacción técnica. |
| Churn_400 | str | 0 | 0 | Variable objetivo: Probabilidad de abandono (Y/N). |

### Definiciones Técnicas y Fórmulas
Para garantizar la coherencia analítica, el cálculo de las métricas de rentabilidad debe seguir las definiciones de negocio estandarizadas. El Margen_eur_bruto se calcula considerando el PVP, el margen operativo del modelo y la carga impositiva:

```sql
ROUND(sales.PVP * (c.Margen) * 0.01 * (1 - sales.IMPUESTOS / 100), 2)
```

La rentabilidad final (Margen_eur) se obtiene sustrayendo los costes de venta, las comisiones del distribuidor, las inversiones en marketing y los gastos logísticos de transporte terrestre (Referencia: Costes.csv):

```sql
  ROUND(
    sales.PVP * (c.Margen) * 0.01 * (1 - sales.IMPUESTOS / 100)
    - sales.COSTE_VENTA_NO_IMPUESTOS
    - (c.Margendistribuidor * 0.01 + c.GastosMarketing * 0.01) * sales.PVP * (1 - sales.IMPUESTOS / 100)
    - c.Costetransporte
  , 2)
```

La identificación de la fuga se rige por la lógica temporal de la variable Churn_400, la cual evalúa tanto la inactividad desde la última revisión como el tiempo transcurrido desde la entrega inicial en ausencia de registros de mantenimiento:

```sql
  CASE
    WHEN SAFE_CAST(log.DIAS_DESDE_ULTIMA_REVISION AS INT64) > 400
      THEN 'Y'
    WHEN log.DIAS_DESDE_ULTIMA_REVISION IS NULL
        AND DATE_DIFF(PARSE_DATE('%d/%m/%Y', sales.Sales_Date), PARSE_DATE('%d/%m/%Y', BASE_DATE), DAY) > 400
      THEN 'Y'
    ELSE 'N'
  END
```

Finalmente, como parte de la entrega opcional, se invita al alumno a proponer un cálculo de CLTV (Customer Lifetime Value) que permita priorizar la intensidad de las acciones comerciales y optimizar el retorno de la inversión según el valor económico esperado de cada segmento.