"""
Churn Predictor - Concesionario de Automóviles
Aplicación Streamlit para visualizar modelos y estrategia comercial
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
import lightgbm as lgb

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATAMART_PATH = os.path.join(BASE_DIR, "Data", "Datamart", "datamart_final_v2.csv")
CUSTOMER_DATA_PATH = os.path.join(BASE_DIR, "Data", "DataLake", "customer_data.csv")
NUEVOS_CLIENTES_PATH = os.path.join(BASE_DIR, "Data", "DataLake", "nuevos_clientes.csv")
COSTES_PATH = os.path.join(BASE_DIR, "Data", "DataLake", "Costes.csv")

FEATURES_NUMERICAS = [
    'km_ultima_revision', 'PVP', 'Edad', 'RENTA_MEDIA_ESTIMADA',
    'gasto_relativo', 'Kw', 'Margen_eur_bruto', 'Margen_eur',
    'ENCUESTA_CLIENTE_ZONA_TALLER'
]
FEATURES_BINARIAS = [
    'tiene_queja', 'en_garantia_bin', 'MANTENIMIENTO_GRATUITO',
    'Lead_compra', 'seguro_bateria_bin', 'sin_encuesta', 'origen_internet'
]
FEATURES_CATEGORICAS = [
    'Modelo', 'ZONA', 'FORMA_PAGO', 'TIPO_CARROCERIA',
    'Equipamiento', 'Fuel', 'TRANSMISION_ID'
]
ALL_FEATURES = FEATURES_NUMERICAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS
COLS_DROP = ['Churn_bin', 'Churn_Final', 'Churn_Corregido', 'perfil_cliente']

MODEL_COLORS = {
    'Random Forest': '#10b981',
    'XGBoost': '#8b5cf6',
    'LightGBM': '#f59e0b'
}

# =============================================================================
# CSS PERSONALIZADO - ESTILO MODERNO
# =============================================================================
st.markdown("""
<style>
    /* Tema principal */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 1rem 0;
        letter-spacing: -1px;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-top: -0.8rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Cards de segmentos */
    .segment-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin: 10px 0;
    }

    .segment-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .segment-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 16px;
    }

    /* Badges de riesgo */
    .risk-badge-alto {
        display: inline-block;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(239,68,68,0.4);
    }

    .risk-badge-medio {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(245,158,11,0.4);
    }

    .risk-badge-bajo {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(16,185,129,0.4);
    }

    /* Métricas estilo */
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #334155;
    }

    /* Divider moderno */
    .modern-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================
@st.cache_data(show_spinner="⏳ Cargando datos...")
def cargar_datos_entrenamiento():
    """Carga datamart y construye Churn_Final."""
    dm = pd.read_csv(DATAMART_PATH)
    df_raw = pd.read_csv(CUSTOMER_DATA_PATH)

    # Construir Churn_Final
    df_raw['Sales_Date'] = pd.to_datetime(df_raw['Sales_Date'], format='%d/%m/%Y', errors='coerce')
    fecha_corte = pd.to_datetime('31/12/2023', format='%d/%m/%Y')
    df_raw['antiguedad_dias'] = (fecha_corte - df_raw['Sales_Date']).dt.days.clip(lower=0)
    df_raw['Churn_bin'] = (df_raw['Churn_400'] == 'Y').astype(int)

    dm['Churn_Final'] = (
        (df_raw['Churn_bin'].values == 1) |
        ((df_raw['Revisiones'].values == 0) & (df_raw['antiguedad_dias'].values > 400))
    ).astype(int)

    cols_drop = [c for c in COLS_DROP if c in dm.columns]
    X = dm.drop(columns=cols_drop)
    y = dm['Churn_Final']

    # Label encoders
    label_encoders = {}
    for col in FEATURES_CATEGORICAS:
        le = LabelEncoder()
        le.fit(df_raw[col].astype(str))
        label_encoders[col] = le

    return X, y, label_encoders

@st.cache_data(show_spinner="⏳ Cargando nuevos clientes...")
def cargar_datos_fresh():
    return pd.read_csv(NUEVOS_CLIENTES_PATH)

@st.cache_data(show_spinner="⏳ Cargando datos de costes...")
def cargar_costes():
    return pd.read_csv(COSTES_PATH)

# =============================================================================
# ENTRENAMIENTO DE MODELOS
# =============================================================================
@st.cache_resource(show_spinner="🤖 Entrenando modelos...")
def entrenar_modelos(_X, _y):
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    modelos = {}

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    modelos['Random Forest'] = _build_model_results(rf, X_test, y_test, X_train.columns)

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=3, min_child_weight=10,
        gamma=1.0, reg_alpha=1.0, reg_lambda=5.0,
        learning_rate=0.05, scale_pos_weight=ratio,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    modelos['XGBoost'] = _build_model_results(xgb_model, X_test, y_test, X_train.columns)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        scale_pos_weight=ratio, random_state=42,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
        n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    modelos['LightGBM'] = _build_model_results(lgb_model, X_test, y_test, X_train.columns)

    return modelos, X_train, X_test, y_train, y_test

def _build_model_results(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cm': confusion_matrix(y_test, y_pred),
        'fpr': fpr,
        'tpr': tpr,
        'prec_curve': prec_curve,
        'rec_curve': rec_curve,
        'metrics': {
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_prob),
            'Avg Precision': average_precision_score(y_test, y_prob),
        },
        'importance': pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
    }

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# =============================================================================
def preprocesar_fresh(fresh_raw, label_encoders, train_columns):
    fresh = fresh_raw.copy()

    fresh['QUEJA'] = fresh['QUEJA'].fillna('NO')
    fresh['tiene_queja'] = (fresh['QUEJA'].str.upper().str.strip() == 'SI').astype(int)
    fresh['en_garantia_bin'] = (fresh['EN_GARANTIA'].fillna('NO').str.upper().str.strip() == 'SI').astype(int)
    fresh['gasto_relativo'] = fresh['PVP'] / fresh['RENTA_MEDIA_ESTIMADA'].clip(lower=1)
    fresh['seguro_bateria_bin'] = (fresh['SEGURO_BATERIA_LARGO_PLAZO'].fillna('NO').str.upper().str.strip() == 'SI').astype(int)
    fresh['sin_encuesta'] = fresh['ENCUESTA_CLIENTE_ZONA_TALLER'].isnull().astype(int)
    fresh['ENCUESTA_CLIENTE_ZONA_TALLER'] = fresh['ENCUESTA_CLIENTE_ZONA_TALLER'].fillna(0)
    fresh['origen_internet'] = (fresh['Origen'].fillna('Tienda').str.strip() == 'Internet').astype(int)

    fresh_feat = fresh[ALL_FEATURES].copy()

    for col in FEATURES_NUMERICAS:
        if col == 'ENCUESTA_CLIENTE_ZONA_TALLER':
            continue
        if fresh_feat[col].isnull().sum() > 0:
            fresh_feat[col] = fresh_feat[col].fillna(fresh_feat[col].median())

    for col in FEATURES_CATEGORICAS:
        if fresh_feat[col].isnull().sum() > 0:
            fresh_feat[col] = fresh_feat[col].fillna(fresh_feat[col].mode()[0])
        le = label_encoders[col]
        known = set(le.classes_)
        fresh_feat[col] = fresh_feat[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    fresh_aligned = fresh_feat.reindex(columns=train_columns, fill_value=0)
    meta = fresh[['CODE', 'Customer_ID', 'Modelo', 'ZONA', 'PVP', 'Edad', 'Revisiones']].copy()

    return fresh_aligned, meta

def construir_input_simulador(valores, label_encoders, train_columns):
    row = {}

    for col in ['PVP', 'Edad', 'RENTA_MEDIA_ESTIMADA', 'Kw',
                'Margen_eur_bruto', 'Margen_eur',
                'ENCUESTA_CLIENTE_ZONA_TALLER', 'km_ultima_revision']:
        row[col] = valores[col]

    row['gasto_relativo'] = row['PVP'] / max(row['RENTA_MEDIA_ESTIMADA'], 1)
    row['sin_encuesta'] = 1 if row['ENCUESTA_CLIENTE_ZONA_TALLER'] == 0 else 0

    for col in ['tiene_queja', 'en_garantia_bin', 'MANTENIMIENTO_GRATUITO',
                'Lead_compra', 'seguro_bateria_bin', 'origen_internet']:
        row[col] = valores[col]

    for col in FEATURES_CATEGORICAS:
        le = label_encoders[col]
        val_str = str(valores[col])
        if val_str in set(le.classes_):
            row[col] = le.transform([val_str])[0]
        else:
            row[col] = -1

    df = pd.DataFrame([row])
    return df.reindex(columns=train_columns, fill_value=0)

def segmento_riesgo(prob):
    if prob >= 0.80: return 'MUY_ALTO'
    if prob >= 0.60: return 'ALTO'
    if prob >= 0.40: return 'MEDIO'
    if prob >= 0.20: return 'BAJO'
    return 'MUY_BAJO'

# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================
def plot_confusion_matrix(cm, model_name, color):
    labels_x = ['Pred: No Churn', 'Pred: Churn']
    labels_y = ['Real: No Churn', 'Real: Churn']
    tn, fp, fn, tp = cm.ravel()
    text_matrix = [[f"{tn:,}\nTN", f"{fp:,}\nFP"], [f"{fn:,}\nFN", f"{tp:,}\nTP"]]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels_x, y=labels_y,
        text=text_matrix, texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        colorscale=[[0, '#0f172a'], [1, color]],
        showscale=False
    ))
    fig.update_layout(
        title=f"<b>{model_name}</b>",
        height=400,
        template="plotly_dark",
        margin=dict(t=50, b=40, l=60, r=40),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    return fig

def plot_roc_curve(model_data, model_name, color):
    auc_val = model_data['metrics']['AUC-ROC']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name='Aleatorio',
        line=dict(color='rgba(148,163,184,0.4)', dash='dash', width=2),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=model_data['fpr'], y=model_data['tpr'],
        name=f"AUC = {auc_val:.4f}",
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.1])}'
    ))
    fig.update_layout(
        title=f"<b>Curva ROC - {model_name}</b>",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        template="plotly_dark",
        legend=dict(x=0.6, y=0.1, bgcolor='rgba(15,23,42,0.8)'),
        margin=dict(l=60, r=40, t=50, b=60)
    )
    return fig

def plot_feature_importance(importance, model_name, color, top_n=10):
    top = importance.head(top_n).sort_values()

    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation='h',
        marker_color=color,
        marker=dict(line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f"{v:.4f}" for v in top.values],
        textposition='outside',
        textfont=dict(size=10, color='white')
    ))
    fig.update_layout(
        title=f"<b>Top {top_n} Variables - {model_name}</b>",
        xaxis_title="Importancia",
        height=450,
        template="plotly_dark",
        margin=dict(l=180, t=50, r=80, b=60),
        xaxis=dict(range=[0, top.values.max() * 1.3])
    )
    return fig

def plot_precision_recall(modelos):
    """Curva Precision-Recall comparativa"""
    fig = go.Figure()
    for name, data in modelos.items():
        ap = data['metrics']['Avg Precision']
        fig.add_trace(go.Scatter(
            x=data['rec_curve'], y=data['prec_curve'],
            name=f"{name} (AP={ap:.4f})",
            line=dict(color=MODEL_COLORS[name], width=3)
        ))
    fig.update_layout(
        title="<b>Curva Precision-Recall</b>",
        xaxis_title="Recall", yaxis_title="Precision",
        height=400, template="plotly_dark",
        legend=dict(x=0.05, y=0.05, bgcolor='rgba(15,23,42,0.8)'),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        margin=dict(l=60, r=40, t=50, b=60)
    )
    return fig

def plot_bimodal_distribution(y_prob, y_test, model_name):
    """Distribución bimodal (camello) de probabilidades"""
    churn_probs = y_prob[y_test.values == 1]
    no_churn_probs = y_prob[y_test.values == 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=no_churn_probs, nbinsx=50, name='No Churn (real)',
        marker_color='#3b82f6', opacity=0.7,
        histnorm='probability density'
    ))
    fig.add_trace(go.Histogram(
        x=churn_probs, nbinsx=50, name='Churn (real)',
        marker_color='#ef4444', opacity=0.7,
        histnorm='probability density'
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="white", line_width=2,
                  annotation_text="Umbral 0.5", annotation_font_color="white")
    fig.update_layout(
        title=f"<b>Distribución de Probabilidades - {model_name}</b>",
        xaxis_title="Probabilidad de Churn",
        yaxis_title="Densidad",
        barmode='overlay',
        height=420, template="plotly_dark",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(15,23,42,0.8)'),
        margin=dict(l=60, r=40, t=50, b=60)
    )
    return fig

def plot_churn_gauge(probability):
    if probability >= 0.80:
        bar_color = '#ef4444'
    elif probability >= 0.60:
        bar_color = '#f59e0b'
    elif probability >= 0.40:
        bar_color = '#06b6d4'
    elif probability >= 0.20:
        bar_color = '#10b981'
    else:
        bar_color = '#10b981'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': 'white', 'family': 'Arial'}},
        title={'text': "<b>Probabilidad de Churn</b>", 'font': {'size': 16, 'color': '#94a3b8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#334155'},
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': '#0f172a',
            'bordercolor': '#1e293b',
            'borderwidth': 2,
            'steps': [
                {'range': [0, 20], 'color': 'rgba(16,185,129,0.15)'},
                {'range': [20, 40], 'color': 'rgba(16,185,129,0.25)'},
                {'range': [40, 60], 'color': 'rgba(6,182,212,0.25)'},
                {'range': [60, 80], 'color': 'rgba(245,158,11,0.25)'},
                {'range': [80, 100], 'color': 'rgba(239,68,68,0.25)'}
            ],
        }
    ))
    fig.update_layout(
        height=300,
        template="plotly_dark",
        margin=dict(t=80, b=20, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_risk_distribution(df):
    """Distribución de riesgo en datos fresh"""
    fig = go.Figure()
    for riesgo, color in [('MUY_BAJO', '#10b981'), ('BAJO', '#84cc16'), ('MEDIO', '#f59e0b'),
                          ('ALTO', '#f97316'), ('MUY_ALTO', '#ef4444')]:
        mask = df['riesgo'] == riesgo
        if mask.any():
            fig.add_trace(go.Histogram(
                x=df.loc[mask, 'prob_churn'], name=riesgo,
                marker_color=color, opacity=0.8,
                nbinsx=30
            ))
    fig.add_vline(x=0.8, line_dash="dash", line_color="#ef4444", line_width=2,
                  annotation_text="Muy Alto (80%)", annotation_font_color="#ef4444")
    fig.add_vline(x=0.6, line_dash="dash", line_color="#f97316", line_width=2,
                  annotation_text="Alto (60%)", annotation_font_color="#f97316")
    fig.add_vline(x=0.4, line_dash="dash", line_color="#f59e0b", line_width=2,
                  annotation_text="Medio (40%)", annotation_font_color="#f59e0b")
    fig.update_layout(
        title="<b>Distribución de Probabilidad de Churn</b>",
        xaxis_title="Probabilidad de Churn",
        yaxis_title="Número de Clientes",
        barmode='overlay',
        height=430, template="plotly_dark",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(15,23,42,0.8)'),
        margin=dict(l=60, r=40, t=50, b=60)
    )
    return fig

def plot_economic_value(df):
    """Valor económico en riesgo por segmento"""
    pvp_by_risk = df.groupby('riesgo')['PVP'].agg(['sum', 'count']).reindex(
        ['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO']
    )
    pvp_millions = pvp_by_risk['sum'] / 1_000_000
    counts = pvp_by_risk['count']

    colors_map = {'MUY_BAJO': '#10b981', 'BAJO': '#84cc16', 'MEDIO': '#f59e0b',
                  'ALTO': '#f97316', 'MUY_ALTO': '#ef4444'}

    fig = go.Figure(go.Bar(
        x=pvp_millions.index,
        y=pvp_millions.values,
        marker_color=[colors_map[s] for s in pvp_millions.index],
        text=[f"{v:,.1f}M€\n({int(c):,} clientes)"
              for v, c in zip(pvp_millions.values, counts.values)],
        textposition='outside',
        textfont=dict(size=12, color='white')
    ))
    fig.update_layout(
        title="<b>Valor Económico en Riesgo (PVP)</b>",
        xaxis_title="Segmento de Riesgo",
        yaxis_title="PVP Total (Millones €)",
        height=430, template="plotly_dark",
        yaxis=dict(range=[0, pvp_millions.max() * 1.35 if pvp_millions.max() > 0 else 1]),
        margin=dict(l=60, r=40, t=50, b=60)
    )
    return fig

def plot_value_vs_risk(df):
    """Matriz estratégica: Valor vs Riesgo"""
    umbral_pvp = df['PVP'].median()
    umbral_churn = 0.5

    def cuadrante(row):
        if row['PVP'] >= umbral_pvp and row['prob_churn'] >= umbral_churn:
            return 'VIPs en Peligro'
        elif row['PVP'] >= umbral_pvp:
            return 'VIPs Seguros'
        elif row['prob_churn'] >= umbral_churn:
            return 'Bajo Valor + Alto Riesgo'
        return 'Bajo Valor + Bajo Riesgo'

    df = df.copy()
    df['cuadrante'] = df.apply(cuadrante, axis=1)

    cuad_colors = {
        'VIPs en Peligro': '#ef4444',
        'VIPs Seguros': '#10b981',
        'Bajo Valor + Alto Riesgo': '#f59e0b',
        'Bajo Valor + Bajo Riesgo': '#3b82f6'
    }

    fig = go.Figure()
    for cuad, color in cuad_colors.items():
        mask = df['cuadrante'] == cuad
        n = mask.sum()
        if n > 0:
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'PVP'],
                y=df.loc[mask, 'prob_churn'],
                mode='markers',
                marker=dict(color=color, size=5, opacity=0.6),
                name=f"{cuad} ({n:,})"
            ))

    fig.add_hline(y=umbral_churn, line_dash="dash", line_color="white", opacity=0.5, line_width=2)
    fig.add_vline(x=umbral_pvp, line_dash="dash", line_color="white", opacity=0.5, line_width=2)

    # Anotación VIPs en peligro
    n_vip_peligro = (df['cuadrante'] == 'VIPs en Peligro').sum()
    pvp_vip = df.loc[df['cuadrante'] == 'VIPs en Peligro', 'PVP'].sum() / 1_000_000
    if n_vip_peligro > 0:
        fig.add_annotation(
            x=df['PVP'].max() * 0.8, y=df['prob_churn'].max() * 0.95,
            text=f"<b>VIPs EN PELIGRO</b><br>{n_vip_peligro:,} clientes<br>{pvp_vip:,.1f}M€",
            showarrow=False, font=dict(size=13, color='#ef4444'),
            bgcolor='rgba(15,23,42,0.9)', bordercolor='#ef4444', borderwidth=2
        )

    fig.update_layout(
        title="<b>Matriz Estratégica: Valor vs Riesgo</b>",
        xaxis_title="PVP del Vehículo (€)",
        yaxis_title="Probabilidad de Churn",
        height=500, template="plotly_dark",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(15,23,42,0.8)'),
        margin=dict(l=60, r=40, t=50, b=60)
    )
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## 🚗 Churn Predictor Pro")
st.sidebar.markdown("### Estrategia de Retención")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "📍 Navegación",
    ["📊 Dashboard & Simulador", "📈 Análisis Nuevos Clientes", "💰 Acción Comercial"],
    index=0
)

# Cargar datos
X, y, label_encoders = cargar_datos_entrenamiento()
modelos, X_train, X_test, y_train, y_test = entrenar_modelos(X, y)

st.sidebar.markdown("---")
st.sidebar.markdown("**Enfoque 2 - Sin Leakage**")
st.sidebar.markdown(f"Clientes: **{len(X):,}**")
st.sidebar.markdown(f"Features: **{X.shape[1]}**")

# =============================================================================
# PÁGINA 1: DASHBOARD + SIMULADOR
# =============================================================================
if pagina == "📊 Dashboard & Simulador":
    st.markdown('<p class="main-header">Dashboard de Modelos & Simulador</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis predictivo con XGBoost, Random Forest y LightGBM</p>', unsafe_allow_html=True)

    # Selector de modelo
    modelo_sel = st.selectbox(
        "🎯 Selecciona modelo:",
        list(modelos.keys()),
        index=1
    )
    color_sel = MODEL_COLORS[modelo_sel]
    data_sel = modelos[modelo_sel]

    # Métricas
    st.markdown("### 📈 Métricas de Rendimiento")
    cols_met = st.columns(5)
    metric_names = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Avg Precision']
    for i, metric in enumerate(metric_names):
        val = data_sel['metrics'][metric]
        cols_met[i].metric(metric, f"{val:.4f}")

    # Tabla comparativa (arriba, justo después de métricas)
    st.markdown("#### 📊 Comparativa de Modelos")
    comp_data = {}
    for name, data in modelos.items():
        comp_data[name] = {m: f"{data['metrics'][m]:.4f}" for m in metric_names}
    comp_df = pd.DataFrame(comp_data).T
    st.dataframe(comp_df, use_container_width=True, height=160)

    st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

    # Fila 1: Confusion Matrix + ROC
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_confusion_matrix(data_sel['cm'], modelo_sel, color_sel), use_container_width=True)
    with col2:
        st.plotly_chart(plot_roc_curve(data_sel, modelo_sel, color_sel), use_container_width=True)

    # Fila 2: Feature Importance + Precision-Recall
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_feature_importance(data_sel['importance'], modelo_sel, color_sel), use_container_width=True)
    with col4:
        st.plotly_chart(plot_precision_recall(modelos), use_container_width=True)

    # Distribución bimodal (camello) - en horizontal
    st.markdown("### 📊 Distribución de Probabilidades")
    st.plotly_chart(plot_bimodal_distribution(data_sel['y_prob'], y_test, modelo_sel), use_container_width=True)

    st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

    # =============================================================================
    # SIMULADOR DE RIESGO (DENTRO DEL DASHBOARD)
    # =============================================================================
    st.markdown("### 🎯 Simulador de Riesgo Individual")
    st.markdown("**Introduce los datos de un cliente para predecir su probabilidad de churn**")

    # Inputs en columnas
    col_inputs1, col_inputs2, col_inputs3 = st.columns(3)

    with col_inputs1:
        st.markdown("**🚗 Vehículo**")
        cats_modelo = sorted(label_encoders['Modelo'].classes_.tolist())
        sim_modelo = st.selectbox("Modelo:", cats_modelo, key='sim_modelo')
        sim_pvp = st.slider("PVP (€):", 10000, 40000, 23500, 500, key='sim_pvp')
        sim_kw = st.slider("Potencia (Kw):", 40, 200, 94, key='sim_kw')
        sim_km = st.slider("Km última revisión:", 0, 100000, 0, 1000, key='sim_km')

    with col_inputs2:
        st.markdown("**👤 Cliente**")
        sim_edad = st.slider("Edad:", 18, 85, 43, key='sim_edad')
        sim_renta = st.slider("Renta media (€):", 0, 40000, 22000, 500, key='sim_renta')
        cats_zona = sorted(label_encoders['ZONA'].classes_.tolist())
        sim_zona = st.selectbox("Zona:", cats_zona, key='sim_zona')
        sim_encuesta = st.slider("Encuesta taller:", 0, 300, 65, key='sim_encuesta')

    with col_inputs3:
        st.markdown("**💰 Comercial**")
        sim_margen_b = st.slider("Margen Bruto (€):", 1000, 13000, 6400, 100, key='sim_mb')
        sim_margen = st.slider("Margen Neto (€):", -11000, 9000, 1700, 100, key='sim_m')
        sim_queja = int(st.checkbox("Tiene queja", key='sim_q'))
        sim_garantia = int(st.checkbox("En garantía", value=True, key='sim_g'))
        sim_mant = st.selectbox("Mant. Gratuito:", [0, 4], key='sim_mant')

    # Construir input y predecir
    valores = {
        'PVP': sim_pvp, 'Edad': sim_edad, 'RENTA_MEDIA_ESTIMADA': sim_renta,
        'Kw': sim_kw, 'Margen_eur_bruto': sim_margen_b, 'Margen_eur': sim_margen,
        'ENCUESTA_CLIENTE_ZONA_TALLER': sim_encuesta, 'km_ultima_revision': sim_km,
        'tiene_queja': sim_queja, 'en_garantia_bin': sim_garantia,
        'MANTENIMIENTO_GRATUITO': sim_mant, 'Lead_compra': 0,
        'seguro_bateria_bin': 0, 'origen_internet': 0,
        'Modelo': sim_modelo, 'ZONA': sim_zona, 'FORMA_PAGO': 'CONTADO',
        'TIPO_CARROCERIA': 'BERLINA', 'Equipamiento': 'MEDIA',
        'Fuel': 'GASOLINA', 'TRANSMISION_ID': 'MANUAL'
    }

    input_df = construir_input_simulador(valores, label_encoders, X_train.columns.tolist())
    model_obj = modelos[modelo_sel]['model']
    prob = model_obj.predict_proba(input_df)[0][1]
    riesgo = segmento_riesgo(prob)

    st.markdown("---")

    # Resultado
    col_gauge, col_result = st.columns([1, 1.5])

    with col_gauge:
        st.plotly_chart(plot_churn_gauge(prob), use_container_width=True)

    with col_result:
        st.markdown("<br>", unsafe_allow_html=True)
        if riesgo == 'MUY_ALTO':
            st.markdown(f'<div class="risk-badge-alto">🔴 RIESGO MUY ALTO - {prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.error("**Acción:** Contacto urgente + Bono 20€ + Lavado completo")
        elif riesgo == 'ALTO':
            st.markdown(f'<div class="risk-badge-alto">🟠 RIESGO ALTO - {prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.warning("**Acción:** Estrategia Premium VIP - Bono 100€ + Recogida + Lavado + Neumáticos")
        elif riesgo == 'MEDIO':
            st.markdown(f'<div class="risk-badge-medio">🟡 RIESGO MEDIO - {prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.info("**Acción:** SMS + Bono 50€ + Regalo + Lavado")
        elif riesgo == 'BAJO':
            st.markdown(f'<div class="risk-badge-bajo">🟢 RIESGO BAJO - {prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.success("**Acción:** SMS + Bono 35€ + Paquete 5 revisiones")
        else:
            st.markdown(f'<div class="risk-badge-bajo">✅ RIESGO MUY BAJO - {prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.success("**Acción:** Email + Bono 10€ + Paquete 5 revisiones")

# =============================================================================
# PÁGINA 2: ANÁLISIS NUEVOS CLIENTES
# =============================================================================
elif pagina == "📈 Análisis Nuevos Clientes":
    st.markdown('<p class="main-header">Predicción - 10,000 Nuevos Clientes</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis de riesgo sobre clientes sin historial de taller</p>', unsafe_allow_html=True)

    # Cargar y procesar fresh
    fresh_raw = cargar_datos_fresh()
    fresh_aligned, fresh_meta = preprocesar_fresh(fresh_raw, label_encoders, X_train.columns.tolist())

    # Selector de modelo
    modelo_fresh_name = st.selectbox(
        "🎯 Modelo de predicción:",
        list(modelos.keys()),
        index=1,
        key='fresh_ml'
    )
    model_fresh = modelos[modelo_fresh_name]['model']

    # Predecir
    fresh_probs = model_fresh.predict_proba(fresh_aligned)[:, 1]
    fresh_preds = model_fresh.predict(fresh_aligned)

    # Resultados
    resultados = fresh_meta.copy()
    resultados['prob_churn'] = fresh_probs
    resultados['pred_churn'] = fresh_preds
    resultados['riesgo'] = resultados['prob_churn'].apply(segmento_riesgo)

    # Filtros
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Filtros")
    zonas = sorted(resultados['ZONA'].unique())
    zonas_sel = st.sidebar.multiselect("Filtrar por ZONA:", zonas, default=zonas)
    modelos_disp = sorted(resultados['Modelo'].unique())
    modelos_sel = st.sidebar.multiselect("Filtrar por MODELO:", modelos_disp, default=modelos_disp)

    # Aplicar filtros
    mask = (resultados['ZONA'].isin(zonas_sel)) & (resultados['Modelo'].isin(modelos_sel))
    res = resultados[mask]

    # KPIs
    n_total = len(res)
    n_muy_alto = (res['riesgo'] == 'MUY_ALTO').sum()
    n_alto = (res['riesgo'] == 'ALTO').sum()
    n_medio = (res['riesgo'] == 'MEDIO').sum()
    n_bajo = (res['riesgo'] == 'BAJO').sum()
    n_muy_bajo = (res['riesgo'] == 'MUY_BAJO').sum()
    pvp_riesgo_alto = res.loc[res['riesgo'].isin(['ALTO', 'MUY_ALTO']), 'PVP'].sum()

    st.markdown("### 📊 Resumen Global")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total", f"{n_total:,}")
    c2.metric("🔴 Muy Alto", f"{n_muy_alto:,}", f"{n_muy_alto/max(n_total,1)*100:.1f}%")
    c3.metric("🟠 Alto", f"{n_alto:,}", f"{n_alto/max(n_total,1)*100:.1f}%")
    c4.metric("🟡 Medio", f"{n_medio:,}", f"{n_medio/max(n_total,1)*100:.1f}%")
    c5.metric("🟢 Bajo", f"{n_bajo:,}", f"{n_bajo/max(n_total,1)*100:.1f}%")
    c6.metric("💰 PVP Riesgo", f"{pvp_riesgo_alto/1_000_000:,.1f}M€")

    st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

    # Gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_risk_distribution(res), use_container_width=True)
    with col2:
        st.plotly_chart(plot_economic_value(res), use_container_width=True)

    st.plotly_chart(plot_value_vs_risk(res), use_container_width=True)

    # Conclusiones
    st.markdown("---")
    st.markdown("###  Conclusiones del Análisis")

    prob_media = res['prob_churn'].mean() * 100 if n_total > 0 else 0
    prob_mediana = res['prob_churn'].median() * 100 if n_total > 0 else 0

    # VIPs en peligro
    umbral_pvp = res['PVP'].median() if n_total > 0 else 0
    vip_peligro = res[(res['PVP'] >= umbral_pvp) & (res['prob_churn'] >= 0.5)]
    n_vip = len(vip_peligro)
    pvp_vip = vip_peligro['PVP'].sum() / 1_000_000

    # Top foco crítico
    if n_total > 0:
        combo = res.groupby(['Modelo', 'ZONA']).agg(
            prob_media_combo=('prob_churn', 'mean'),
            n_combo=('prob_churn', 'count')
        ).sort_values('prob_media_combo', ascending=False)
        if len(combo) > 0:
            top_combo = combo.iloc[0]
            top_modelo, top_zona = combo.index[0]
            top_prob = top_combo['prob_media_combo'] * 100
        else:
            top_modelo, top_zona, top_prob = '-', '-', 0
    else:
        top_modelo, top_zona, top_prob = '-', '-', 0

    st.info(f"""
**Resumen Ejecutivo — {n_total:,} clientes analizados (modelo: {modelo_fresh_name})**

- **Probabilidad media de churn:** {prob_media:.1f}% | **Mediana:** {prob_mediana:.1f}%
- **Riesgo MUY ALTO (>80%):** {n_muy_alto:,} clientes ({n_muy_alto/max(n_total,1)*100:.1f}%)
- **Riesgo ALTO (60-80%):** {n_alto:,} clientes ({n_alto/max(n_total,1)*100:.1f}%)
- **Riesgo MEDIO (40-60%):** {n_medio:,} clientes ({n_medio/max(n_total,1)*100:.1f}%)
- **Riesgo BAJO (20-40%):** {n_bajo:,} clientes ({n_bajo/max(n_total,1)*100:.1f}%)
- **Riesgo MUY BAJO (<20%):** {n_muy_bajo:,} clientes ({n_muy_bajo/max(n_total,1)*100:.1f}%)
- **VIPs en peligro** (alto valor + alta prob. churn): **{n_vip:,} clientes** — **{pvp_vip:,.1f}M€**
- **Foco crítico:** Modelo **{top_modelo}** en zona **{top_zona}** ({top_prob:.1f}% churn medio)

**Nota:** La variable `perfil_cliente` fue excluida del modelo porque todos los nuevos clientes son Ghost (sin visitas al taller) por definición. El modelo usa `Churn_Final` (33.3%) como target.
    """)

    # Tabla expandible
    with st.expander("Ver tabla de resultados completa"):
        st.dataframe(
            res.sort_values('prob_churn', ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=400
        )

# =============================================================================
# PÁGINA 3: ACCIÓN COMERCIAL
# =============================================================================
# PÁGINA 3: ACCIÓN COMERCIAL
# =============================================================================
else:
    st.markdown('<p class="main-header">Estrategia de Retención Comercial</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis ROI y plan de acción por segmento · 10,000 nuevos clientes</p>', unsafe_allow_html=True)

    # ── Cargar y preparar datos ────────────────────────────────────────────────
    fresh_raw    = cargar_datos_fresh()
    costes       = cargar_costes()
    fresh_aligned, fresh_meta = preprocesar_fresh(fresh_raw, label_encoders, X_train.columns.tolist())

    model_xgb    = modelos['XGBoost']['model']
    fresh_probs  = model_xgb.predict_proba(fresh_aligned)[:, 1]

    clientes = fresh_meta.copy()
    clientes['prob_churn'] = fresh_probs
    clientes['segmento']   = clientes['prob_churn'].apply(segmento_riesgo)
    clientes = clientes.merge(costes[['Modelo', 'Mantenimiento_medio', 'Margen']], on='Modelo', how='left')

    def calcular_cf(row):
        base  = row['Mantenimiento_medio']
        alpha = 0.07 if row['Modelo'] in ['A', 'B'] else 0.10
        n     = row['Revisiones']
        return base * ((1 + alpha) ** (n + 1))

    clientes['CF']              = clientes.apply(calcular_cf, axis=1)
    clientes['Precio']          = clientes['CF'] / (clientes['Margen'] / 100)
    clientes['Margen_Bruto']    = clientes['Precio'] - clientes['CF']
    clientes['Coste_Marketing'] = clientes['CF'] * 0.01

    COSTE_CONTACTO  = {'MUY_ALTO': 0.20, 'ALTO': 0.10, 'MEDIO': 0.50, 'BAJO': 0.50, 'MUY_BAJO': 0.20}
    COSTE_ADICIONAL = {'MUY_ALTO': 30.00, 'ALTO': 100.00, 'MEDIO': 36.00, 'BAJO': 0.00, 'MUY_BAJO': 0.00}
    RESPONSE_RATE   = {'MUY_ALTO': 0.10, 'ALTO': 0.80, 'MEDIO': 0.55, 'BAJO': 0.45, 'MUY_BAJO': 0.40}
    BONOS_FIJOS     = {'MUY_ALTO': 20.00, 'ALTO': 100.00, 'MEDIO': 50.00, 'BAJO': 35.00, 'MUY_BAJO': 10.00}

    clientes['Bono_€']          = clientes['segmento'].map(BONOS_FIJOS)
    clientes['RR']              = clientes['segmento'].map(RESPONSE_RATE)
    clientes['Coste_Contacto']  = clientes['segmento'].map(COSTE_CONTACTO)
    clientes['Coste_Adicional'] = clientes['segmento'].map(COSTE_ADICIONAL)

    # ── Cálculo ROI por segmento ───────────────────────────────────────────────
    resultados_roi = []
    for segmento in ['MUY_ALTO', 'ALTO', 'MEDIO', 'BAJO', 'MUY_BAJO']:
        seg = clientes[clientes['segmento'] == segmento]
        if len(seg) == 0:
            continue
        n_total     = len(seg)
        rr          = RESPONSE_RATE[segmento]
        n_responden = n_total * rr

        cc_total  = n_total     * COSTE_CONTACTO[segmento]
        ca_total  = n_responden * COSTE_ADICIONAL[segmento]
        cm_total  = n_responden * seg['Coste_Marketing'].mean()
        cd_total  = n_responden * seg['Bono_€'].mean()
        inv_total = cc_total + ca_total + cm_total + cd_total

        mb_total = n_responden * seg['Margen_Bruto'].mean()
        bn_total = mb_total - inv_total
        roi      = (bn_total / inv_total) if inv_total > 0 else 0

        resultados_roi.append({
            'Segmento':        segmento,
            'N_Clientes':      n_total,
            'RR_%':            int(rr * 100),
            'N_Captados':      int(n_responden),
            'Inversion_€':     inv_total,
            'Margen_Bruto_€':  mb_total,
            'Beneficio_Neto_€': bn_total,
            'ROI':             roi
        })

    df_roi     = pd.DataFrame(resultados_roi)
    inv_global = df_roi['Inversion_€'].sum()
    bn_global  = df_roi['Beneficio_Neto_€'].sum()
    roi_global = (bn_global / inv_global) if inv_global > 0 else 0
    cap_global = int(df_roi['N_Captados'].sum())

    # ── CLTV: Customer Lifetime Value ─────────────────────────────────────────
    MAX_REV = 20
    prob_aj = clientes['prob_churn'].clip(lower=0.01)
    clientes['CLTV']      = clientes['Margen_Bruto'] * ((1 - prob_aj) / prob_aj).clip(upper=MAX_REV)
    p33 = clientes['CLTV'].quantile(0.33)
    p66 = clientes['CLTV'].quantile(0.66)
    clientes['cltv_nivel'] = clientes['CLTV'].apply(
        lambda x: 'ALTO' if x > p66 else ('MEDIO' if x > p33 else 'BAJO')
    )

    ESTRATEGIA_2D = {
        ('MUY_ALTO', 'BAJO'):  (0.20,   0.00,  10.00, 0.08),
        ('MUY_ALTO', 'MEDIO'): (0.20,  30.00,  20.00, 0.10),
        ('MUY_ALTO', 'ALTO'):  (0.20,  30.00,  50.00, 0.15),
        ('ALTO',     'BAJO'):  (0.50,   0.00,  30.00, 0.40),
        ('ALTO',     'MEDIO'): (0.10, 100.00, 100.00, 0.80),
        ('ALTO',     'ALTO'):  (0.10, 100.00, 150.00, 0.85),
        ('MEDIO',    'BAJO'):  (0.50,   0.00,  15.00, 0.30),
        ('MEDIO',    'MEDIO'): (0.50,  36.00,  50.00, 0.55),
        ('MEDIO',    'ALTO'):  (0.50,  36.00,  80.00, 0.60),
        ('BAJO',     'BAJO'):  (0.20,   0.00,   0.00, 0.10),
        ('BAJO',     'MEDIO'): (0.50,   0.00,  35.00, 0.45),
        ('BAJO',     'ALTO'):  (0.50,   0.00,  35.00, 0.50),
        ('MUY_BAJO', 'BAJO'):  (0.00,   0.00,   0.00, 0.00),
        ('MUY_BAJO', 'MEDIO'): (0.20,   0.00,  10.00, 0.40),
        ('MUY_BAJO', 'ALTO'):  (0.20,   0.00,  10.00, 0.45),
    }

    res2d = clientes.apply(
        lambda r: ESTRATEGIA_2D.get((r['segmento'], r['cltv_nivel']), (0.20, 0.00, 10.00, 0.20)),
        axis=1, result_type='expand'
    )
    res2d.columns = ['CC2', 'CA2', 'Bono2', 'RR2']
    clientes = pd.concat([clientes, res2d], axis=1)

    resultados_cltv = []
    for seg in ['MUY_ALTO', 'ALTO', 'MEDIO', 'BAJO', 'MUY_BAJO']:
        for nivel in ['BAJO', 'MEDIO', 'ALTO']:
            g = clientes[(clientes['segmento'] == seg) & (clientes['cltv_nivel'] == nivel)]
            if len(g) == 0:
                continue
            cc, ca, bono, rr = ESTRATEGIA_2D[(seg, nivel)]
            n  = len(g); nr = n * rr
            inv = n*cc + nr*ca + nr*g['Coste_Marketing'].mean() + nr*bono
            mb  = nr * g['Margen_Bruto'].mean()
            bn  = mb - inv
            resultados_cltv.append({
                'Churn': seg, 'CLTV': nivel, 'N': n,
                'RR_%': int(rr*100), 'N_Captados': int(nr),
                'Inversion_€': inv, 'Margen_Bruto_€': mb,
                'Beneficio_Neto_€': bn, 'ROI': (bn/inv) if inv > 0 else 0
            })

    df_roi_cltv    = pd.DataFrame(resultados_cltv)
    inv_global_c   = df_roi_cltv['Inversion_€'].sum()
    bn_global_c    = df_roi_cltv['Beneficio_Neto_€'].sum()
    roi_global_c   = (bn_global_c / inv_global_c) if inv_global_c > 0 else 0
    cap_global_c   = int(df_roi_cltv['N_Captados'].sum())

    # ROI por segmento agregado (para comparativa)
    roi_cltv_seg = (
        df_roi_cltv.groupby('Churn')['Beneficio_Neto_€'].sum() /
        df_roi_cltv.groupby('Churn')['Inversion_€'].sum()
    )

    # ── CSS adicional ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .seg-header { display:flex; align-items:center; gap:14px; margin-bottom:18px; }
    .seg-badge  { font-size:1.05rem; font-weight:800; padding:6px 18px; border-radius:30px; letter-spacing:0.5px; }
    .seg-desc   { font-size:0.95rem; color:#94a3b8; }
    .econ-table { width:100%; border-collapse:collapse; margin-bottom:20px; font-size:0.92rem; }
    .econ-table th {
        background:#1e293b; color:#94a3b8; font-weight:600; padding:10px 14px;
        text-align:center; border-bottom:2px solid #334155;
        font-size:0.82rem; text-transform:uppercase; letter-spacing:0.5px;
    }
    .econ-table td {
        padding:11px 14px; text-align:center; border-bottom:1px solid #1e293b;
        color:#e2e8f0; font-weight:500;
    }
    .econ-table tr:last-child td { border-bottom:none; }
    .econ-table td.highlight { font-size:1.05rem; font-weight:700; }
    .strat-box  { background:#0f172a; border-radius:12px; padding:20px 24px; border-left:4px solid; margin-top:4px; }
    .strat-title { font-size:1.1rem; font-weight:700; margin-bottom:10px; }
    .strat-item { display:flex; align-items:flex-start; gap:10px; margin:7px 0; font-size:0.93rem; color:#cbd5e1; }
    .strat-icon { font-size:1.1rem; flex-shrink:0; }
    .roi-pill   { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.85rem; font-weight:700; margin-left:8px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Tabs: Solo Churn vs Churn × CLTV ──────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Estrategia Solo Churn", "🎯 Estrategia Churn × CLTV"])

    with tab1:
        # ════════════════════════════════════════════════════════════════════════════
        # BLOQUE 1 · KPIs GLOBALES
        # ════════════════════════════════════════════════════════════════════════════
        st.markdown("### 📊 Resultados Globales de la Campaña")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(" ROI Global",        f"{roi_global:.2f}x")
        k2.metric(" Inversión Total",   f"{inv_global:,.0f} €")
        k3.metric(" Beneficio Neto",    f"{bn_global:,.0f} €")
        k4.metric(" Clientes Captados", f"{cap_global:,}")

        st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════════
        # BLOQUE 2 · ESTRATEGIAS POR SEGMENTO
        # ════════════════════════════════════════════════════════════════════════════
        st.markdown("### 🎯 Estrategia por Segmento")

        SEG_CONFIG = {
            'MUY_ALTO': {
                'color':      '#ef4444',
                'bg':         'rgba(239,68,68,0.08)',
                'emoji':      '🔴',
                'label':      'Riesgo Muy Alto',
                'prob_range': '>80%',
                'estrategia': 'Rescate Urgente',
                'canal':      'Llamada directa del asesor + Email personalizado',
                'oferta':     'Revisión gratuita + Lavado completo + Bono 20€ descuento en próximo servicio',
                'justif':     'Clientes con muy alta probabilidad de abandono pero bajo response rate (10%). La acción debe ser inmediata, directa y de bajo coste por contacto para minimizar pérdidas. El bono moderado equilibra el riesgo de no recuperación.',
                'acciones': [
                    ('', 'Llamada personal del asesor en los próximos 5 días hábiles'),
                    ('', 'Email con oferta exclusiva de revisión gratuita'),
                    ('', 'Lavado completo de cortesía al traer el vehículo'),
                    ('', 'Bono de 20€ aplicable en el siguiente servicio'),
                ]
            },
            'ALTO': {
                'color':      '#f97316',
                'bg':         'rgba(249,115,22,0.08)',
                'emoji':      '🟠',
                'label':      'Riesgo Alto',
                'prob_range': '60-80%',
                'estrategia': 'Retención Premium VIP',
                'canal':      'Email personalizado + SMS de confirmación',
                'oferta':     'Recogida del vehículo + Lavado premium + Cambio de neumáticos con descuento + Bono 100€',
                'justif':     'Segmento de mayor palanca económica: response rate del 80% y bono de 100€ generan el mayor retorno absoluto de la campaña. Es el grupo donde más merece la pena invertir, combinando servicio diferencial y oferta económica significativa.',
                'acciones': [
                    ('', 'Email VIP con propuesta personalizada y nombre del cliente'),
                    ('', 'Servicio de recogida y entrega del vehículo en domicilio'),
                    ('', 'Lavado interior y exterior premium de cortesía'),
                    ('', 'Descuento preferente en cambio de neumáticos'),
                    ('', 'Bono de 100€ aplicable en revisión + opción segundo vehículo (si n≥5)'),
                ]
            },
            'MEDIO': {
                'color':      '#f59e0b',
                'bg':         'rgba(245,158,11,0.08)',
                'emoji':      '🟡',
                'label':      'Riesgo Medio',
                'prob_range': '40-60%',
                'estrategia': 'Fidelización con Incentivo',
                'canal':      'SMS + Email de seguimiento',
                'oferta':     'Lavado gratuito + Regalo sorpresa + Bono 50€',
                'justif':     'Clientes en zona de indecisión. Un incentivo tangible (regalo + bono) con bajo coste de contacto vía SMS genera un response rate del 55% y alta rentabilidad. La sorpresa del regalo actúa como palanca emocional.',
                'acciones': [
                    ('', 'SMS con código de oferta exclusivo y enlace de reserva'),
                    ('', 'Regalo sorpresa al acudir al taller (accesorio de vehículo)'),
                    ('', 'Lavado gratuito incluido en la visita'),
                    ('', 'Bono de 50€ de descuento en el servicio de revisión'),
                    ('', 'Email de seguimiento si no hay respuesta en 7 días'),
                ]
            },
            'BAJO': {
                'color':      '#84cc16',
                'bg':         'rgba(132,204,22,0.08)',
                'emoji':      '🟢',
                'label':      'Riesgo Bajo',
                'prob_range': '20-40%',
                'estrategia': 'Lock-in de Fidelidad',
                'canal':      'SMS o Email (canal preferido del cliente)',
                'oferta':     'Paquete 5 revisiones con -15% + Bono 35€',
                'justif':     'Clientes satisfechos que no requieren rescate urgente. El objetivo es bloquear su fidelidad a largo plazo mediante paquetes prepago ventajosos. El bono de 35€ cubre el coste percibido del compromiso y asegura ingresos futuros.',
                'acciones': [
                    ('', 'SMS/Email con propuesta de paquete de revisiones'),
                    ('', 'Oferta de bono 5 revisiones con descuento del 15%'),
                    ('', 'Bono de 35€ al contratar el paquete'),
                    ('', 'Recordatorio automático antes de fecha de próxima revisión'),
                ]
            },
            'MUY_BAJO': {
                'color':      '#10b981',
                'bg':         'rgba(16,185,129,0.08)',
                'emoji':      '✅',
                'label':      'Riesgo Muy Bajo',
                'prob_range': '<20%',
                'estrategia': 'Mantenimiento de Relación',
                'canal':      'Email informativo',
                'oferta':     'Paquete de revisiones + Descuento por fidelidad + Bono 10€',
                'justif':     'Clientes muy fieles con mínimo riesgo de churn. La inversión debe ser mínima: un email bien diseñado con un pequeño incentivo mantiene el vínculo sin erosionar margen. El objetivo es confirmar la fidelidad, no rescatar.',
                'acciones': [
                    ('', 'Email de agradecimiento por fidelidad con oferta de mantenimiento'),
                    ('', 'Descuento por ser cliente recurrente en próxima revisión'),
                    ('', 'Bono simbólico de 10€ como reconocimiento'),
                    ('', 'Invitación a programa de referidos con incentivo'),
                ]
            },
        }

        for _, row in df_roi.iterrows():
            seg       = row['Segmento']
            cfg       = SEG_CONFIG[seg]
            roi_color = '#10b981' if row['ROI'] >= 3 else '#f59e0b' if row['ROI'] >= 1 else '#ef4444'

            # Cabecera
            st.markdown(f"""
            <div style="margin-top:2.5rem; margin-bottom:0.5rem;">
                <div class="seg-header">
                    <span class="seg-badge" style="background:{cfg['bg']}; color:{cfg['color']}; border:1.5px solid {cfg['color']};">
                        {cfg['emoji']} {cfg['label']}
                    </span>
                    <span class="seg-desc">
                        Probabilidad de churn: <strong style="color:{cfg['color']}">{cfg['prob_range']}</strong>
                        &nbsp;·&nbsp; Estrategia: <strong style="color:#e2e8f0">{cfg['estrategia']}</strong>
                        <span class="roi-pill" style="background:{roi_color}22; color:{roi_color}; border:1px solid {roi_color}">
                            ROI {row['ROI']:.2f}x
                        </span>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Tabla económica
            bn_color = '#10b981' if row['Beneficio_Neto_€'] >= 0 else '#ef4444'
            st.markdown(f"""
            <table class="econ-table">
                <thead>
                    <tr>
                        <th>Clientes</th>
                        <th>Captados</th>
                        <th>Response Rate</th>
                        <th>Inversión</th>
                        <th>Margen Bruto</th>
                        <th>Beneficio Neto</th>
                        <th>ROI</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="background:{cfg['bg']}">
                        <td><strong>{row['N_Clientes']:,}</strong></td>
                        <td><strong>{row['N_Captados']:,}</strong></td>
                        <td><strong>{row['RR_%']}%</strong></td>
                        <td>{row['Inversion_€']:,.0f} €</td>
                        <td>{row['Margen_Bruto_€']:,.0f} €</td>
                        <td class="highlight" style="color:{bn_color}">{row['Beneficio_Neto_€']:,.0f} €</td>
                        <td class="highlight" style="color:{roi_color}">{row['ROI']:.2f}x</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)

            # Estrategia detallada: dos columnas
            col_strat, col_just = st.columns([1.1, 0.9])

            with col_strat:
                acciones_html = "".join([
                    f'<div class="strat-item"><span class="strat-icon">{ico}</span><span>{txt}</span></div>'
                    for ico, txt in cfg['acciones']
                ])
                st.markdown(f"""
                <div class="strat-box" style="border-color:{cfg['color']}">
                    <div class="strat-title" style="color:{cfg['color']}">📋 Plan de Acción</div>
                    <div style="font-size:0.82rem; color:#64748b; margin-bottom:10px;">
                        Canal principal: <strong style="color:#94a3b8">{cfg['canal']}</strong>
                    </div>
                    {acciones_html}
                </div>
                """, unsafe_allow_html=True)

            with col_just:
                st.markdown(f"""
                <div class="strat-box" style="border-color:#334155; margin-top:0">
                    <div class="strat-title" style="color:#94a3b8">💡 Justificación Económica</div>
                    <div style="font-size:0.91rem; color:#cbd5e1; line-height:1.6;">
                        {cfg['justif']}
                    </div>
                    <div style="margin-top:14px; padding-top:12px; border-top:1px solid #1e293b; font-size:0.83rem; color:#64748b;">
                        <strong style="color:#94a3b8">Oferta:</strong> {cfg['oferta']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════════
        # BLOQUE 3 · VISUALIZACIONES
        # ════════════════════════════════════════════════════════════════════════════
        st.markdown("### 📈 Visualizaciones")

        # ── Gauge ROI centrado ─────────────────────────────────────────────────────
        col_l, col_center, col_r = st.columns([1, 2, 1])
        with col_center:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=roi_global,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': "<b>ROI Global</b><br><span style='font-size:13px;color:#94a3b8'>Retorno por € invertido</span>",
                    'font': {'size': 20, 'color': 'white'}
                },
                number={'font': {'size': 64, 'color': '#10b981', 'family': 'Arial Black'}, 'suffix': 'x', 'valueformat': '.2f'},
                delta={'reference': 3.0, 'increasing': {'color': '#10b981'}, 'decreasing': {'color': '#ef4444'}, 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 6], 'tickwidth': 2, 'tickcolor': '#334155'},
                    'bar': {'color': '#10b981', 'thickness': 0.75},
                    'bgcolor': '#0f172a',
                    'borderwidth': 2,
                    'bordercolor': '#1e293b',
                    'steps': [
                        {'range': [0, 1], 'color': 'rgba(239,68,68,0.3)'},
                        {'range': [1, 2], 'color': 'rgba(245,158,11,0.25)'},
                        {'range': [2, 3], 'color': 'rgba(234,179,8,0.2)'},
                        {'range': [3, 4], 'color': 'rgba(132,204,22,0.2)'},
                        {'range': [4, 6], 'color': 'rgba(16,185,129,0.2)'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 4}, 'thickness': 0.8, 'value': 3.0}
                }
            ))
            fig_gauge.update_layout(
                template='plotly_dark',
                height=400,
                margin={'t': 50, 'b': 10, 'l': 30, 'r': 30},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Inversión vs Beneficio  |  Clientes Totales vs Captados ───────────────
        col1, col2 = st.columns(2)

        with col1:
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Bar(
                name='Inversión',
                x=df_roi['Segmento'],
                y=df_roi['Inversion_€'],
                marker_color='#6366f1',
                text=[f"{v/1000:.0f}K€" for v in df_roi['Inversion_€']],
                textposition='outside',
                textfont=dict(size=11, color='white')
            ))
            fig_inv.add_trace(go.Bar(
                name='Beneficio Neto',
                x=df_roi['Segmento'],
                y=df_roi['Beneficio_Neto_€'],
                marker_color=[('#10b981' if v >= 0 else '#ef4444') for v in df_roi['Beneficio_Neto_€']],
                text=[f"{v/1000:.0f}K€" for v in df_roi['Beneficio_Neto_€']],
                textposition='outside',
                textfont=dict(size=11, color='white')
            ))
            fig_inv.update_layout(
                title="<b>Inversión vs Beneficio Neto</b>",
                xaxis_title="Segmento",
                yaxis_title="Euros (€)",
                barmode='group',
                template='plotly_dark',
                height=420,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(15,23,42,0.8)'),
                margin=dict(t=60, b=60, l=40, r=20)
            )
            st.plotly_chart(fig_inv, use_container_width=True)

        with col2:
            fig_funnel = go.Figure()
            fig_funnel.add_trace(go.Bar(
                name='Clientes Totales',
                x=df_roi['Segmento'],
                y=df_roi['N_Clientes'],
                marker_color='#334155',
                text=df_roi['N_Clientes'],
                textposition='inside',
                textfont=dict(size=12, color='white')
            ))
            fig_funnel.add_trace(go.Bar(
                name='Clientes Captados',
                x=df_roi['Segmento'],
                y=df_roi['N_Captados'],
                marker_color=[SEG_CONFIG[s]['color'] for s in df_roi['Segmento']],
                text=df_roi['N_Captados'],
                textposition='inside',
                textfont=dict(size=12, color='white')
            ))
            fig_funnel.update_layout(
                title="<b>Clientes Totales vs Captados</b>",
                xaxis_title="Segmento",
                yaxis_title="Nº Clientes",
                barmode='overlay',
                template='plotly_dark',
                height=420,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(15,23,42,0.8)'),
                margin=dict(t=60, b=60, l=40, r=20)
            )
            st.plotly_chart(fig_funnel, use_container_width=True)

    with tab2:
        # ════════════════════════════════════════════════════════════════════════
        # CLTV TAB · KPIs GLOBALES
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("### 📊 Resultados Globales de la Campaña (Churn × CLTV)")
        st.markdown(
            "<p style='color:#94a3b8;font-size:0.9rem;margin-top:-10px;'>"
            "La oferta se adapta al valor económico esperado de cada cliente. "
            "Clientes con CLTV bajo reciben menor inversión; clientes con CLTV alto reciben ofertas más agresivas.</p>",
            unsafe_allow_html=True
        )

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("ROI Global",        f"{roi_global_c:.2f}x",  delta=f"{roi_global_c - roi_global:+.2f}x vs Solo-Churn")
        kc2.metric("Inversión Total",   f"{inv_global_c:,.0f} €", delta=f"{inv_global_c - inv_global:+,.0f} €")
        kc3.metric("Beneficio Neto",    f"{bn_global_c:,.0f} €",  delta=f"{bn_global_c - bn_global:+,.0f} €")
        kc4.metric("Clientes Captados", f"{cap_global_c:,}",       delta=f"{cap_global_c - cap_global:+,}")

        st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════
        # CLTV TAB · ESTRATEGIAS POR SEGMENTO (matriz 2D)
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("### 🎯 Estrategia por Segmento × CLTV")

        SEG_CONFIG_C = {
            'MUY_ALTO': ('#ef4444', 'rgba(239,68,68,0.08)',  '🔴', 'Riesgo Muy Alto', '>80%'),
            'ALTO'    : ('#f97316', 'rgba(249,115,22,0.08)', '🟠', 'Riesgo Alto',     '60-80%'),
            'MEDIO'   : ('#f59e0b', 'rgba(245,158,11,0.08)', '🟡', 'Riesgo Medio',    '40-60%'),
            'BAJO'    : ('#84cc16', 'rgba(132,204,22,0.08)', '🟢', 'Riesgo Bajo',     '20-40%'),
            'MUY_BAJO': ('#10b981', 'rgba(16,185,129,0.08)', '✅', 'Riesgo Muy Bajo', '<20%'),
        }
        CLTV_ACCIONES = {
            ('MUY_ALTO', 'BAJO') : 'Email básico · Bono 10€',
            ('MUY_ALTO', 'MEDIO'): 'Email + Lavado · Bono 20€',
            ('MUY_ALTO', 'ALTO') : 'Email + Lavado + Bono EXTRA · Bono 50€',
            ('ALTO',     'BAJO') : 'SMS + Bono reducido · Bono 30€',
            ('ALTO',     'MEDIO'): 'Email + Pack Premium · Bono 100€',
            ('ALTO',     'ALTO') : 'Email + Pack Premium VIP · Bono 150€',
            ('MEDIO',    'BAJO') : 'SMS básico · Bono 15€',
            ('MEDIO',    'MEDIO'): 'SMS + Lavado + Bono · Bono 50€',
            ('MEDIO',    'ALTO') : 'SMS + Lavado + Bono EXTRA · Bono 80€',
            ('BAJO',     'BAJO') : 'Email básico (sin bono)',
            ('BAJO',     'MEDIO'): 'SMS + Bono 35€',
            ('BAJO',     'ALTO') : 'SMS + Paquete 5 rev. · Bono 35€',
            ('MUY_BAJO', 'BAJO') : 'Sin acción',
            ('MUY_BAJO', 'MEDIO'): 'Email + Bono 10€',
            ('MUY_BAJO', 'ALTO') : 'Email + Paquete 5 rev. Premium · Bono 10€',
        }

        for seg in ['MUY_ALTO', 'ALTO', 'MEDIO', 'BAJO', 'MUY_BAJO']:
            color, bg, emoji, label, prob_range = SEG_CONFIG_C[seg]
            seg_rows = df_roi_cltv[df_roi_cltv['Churn'] == seg]
            if len(seg_rows) == 0:
                continue
            inv_seg = seg_rows['Inversion_€'].sum()
            roi_seg = seg_rows['Beneficio_Neto_€'].sum() / inv_seg if inv_seg > 0 else 0
            roi_color = '#10b981' if roi_seg >= 3 else '#f59e0b' if roi_seg >= 1 else '#ef4444'

            st.markdown(f"""
            <div style="margin-top:2.5rem; margin-bottom:0.5rem;">
                <div class="seg-header">
                    <span class="seg-badge" style="background:{bg}; color:{color}; border:1.5px solid {color};">
                        {emoji} {label}
                    </span>
                    <span class="seg-desc">
                        Probabilidad de churn: <strong style="color:{color}">{prob_range}</strong>
                        &nbsp;·&nbsp; ROI agregado:
                        <span class="roi-pill" style="background:{roi_color}22; color:{roi_color}; border:1px solid {roi_color}">
                            {roi_seg:.2f}x
                        </span>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            rows_html = ""
            for _, r in seg_rows.sort_values('CLTV').iterrows():
                nivel = r['CLTV']
                rc = '#10b981' if r['ROI'] >= 3 else '#f59e0b' if r['ROI'] >= 1 else '#ef4444'
                accion = CLTV_ACCIONES.get((seg, nivel), '—')
                cltv_color = {'BAJO': '#94a3b8', 'MEDIO': '#f59e0b', 'ALTO': '#10b981'}.get(nivel, 'white')
                bn_c = '#10b981' if r['Beneficio_Neto_€'] >= 0 else '#ef4444'
                rows_html += f"""
                <tr>
                    <td><strong style="color:{cltv_color}">{nivel}</strong></td>
                    <td>{int(r['N']):,}</td>
                    <td>{int(r['N_Captados']):,}</td>
                    <td>{r['RR_%']}%</td>
                    <td>{r['Inversion_€']:,.0f} €</td>
                    <td>{r['Margen_Bruto_€']:,.0f} €</td>
                    <td style="color:{bn_c}">{r['Beneficio_Neto_€']:,.0f} €</td>
                    <td style="color:{rc}; font-weight:700">{r['ROI']:.2f}x</td>
                    <td style="color:#94a3b8; font-size:0.85rem">{accion}</td>
                </tr>"""

            st.markdown(f"""
            <table class="econ-table">
                <thead><tr>
                    <th>CLTV</th><th>Clientes</th><th>Captados</th><th>RR</th>
                    <th>Inversión</th><th>Margen Bruto</th><th>Beneficio Neto</th><th>ROI</th><th>Acción</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════
        # CLTV TAB · VISUALIZACIONES (mismas gráficas + referencia solo-churn)
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("### 📈 Visualizaciones")

        col_l2, col_c2, col_r2 = st.columns([1, 2, 1])
        with col_c2:
            fig_gauge_c = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=roi_global_c,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': "<b>ROI Global · Churn × CLTV</b><br><span style='font-size:13px;color:#94a3b8'>Retorno por € invertido</span>",
                    'font': {'size': 20, 'color': 'white'}
                },
                number={'font': {'size': 64, 'color': '#3b82f6', 'family': 'Arial Black'}, 'suffix': 'x', 'valueformat': '.2f'},
                delta={'reference': roi_global, 'increasing': {'color': '#10b981'}, 'decreasing': {'color': '#ef4444'}, 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 6], 'tickwidth': 2, 'tickcolor': '#334155'},
                    'bar': {'color': '#3b82f6', 'thickness': 0.75},
                    'bgcolor': '#0f172a', 'borderwidth': 2, 'bordercolor': '#1e293b',
                    'steps': [
                        {'range': [0, 1], 'color': 'rgba(239,68,68,0.3)'},
                        {'range': [1, 2], 'color': 'rgba(245,158,11,0.25)'},
                        {'range': [2, 3], 'color': 'rgba(234,179,8,0.2)'},
                        {'range': [3, 4], 'color': 'rgba(132,204,22,0.2)'},
                        {'range': [4, 6], 'color': 'rgba(16,185,129,0.2)'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 4}, 'thickness': 0.8, 'value': 3.0}
                }
            ))
            fig_gauge_c.update_layout(
                template='plotly_dark', height=400,
                margin={'t': 50, 'b': 10, 'l': 30, 'r': 30},
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_gauge_c, use_container_width=True)

        df_roi_cltv_seg = df_roi_cltv.groupby('Churn')[
            ['N', 'N_Captados', 'Inversion_€', 'Margen_Bruto_€', 'Beneficio_Neto_€']
        ].sum().reindex(['MUY_ALTO', 'ALTO', 'MEDIO', 'BAJO', 'MUY_BAJO']).reset_index()
        df_roi_cltv_seg.rename(columns={'Churn': 'Segmento', 'N': 'N_Clientes'}, inplace=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_inv_c = go.Figure()
            fig_inv_c.add_trace(go.Bar(
                name='Inversión (Solo Churn)',
                x=df_roi['Segmento'], y=df_roi['Inversion_€'],
                marker_color='rgba(99,102,241,0.45)',
                text=[f"{v/1000:.0f}K€" for v in df_roi['Inversion_€']],
                textposition='outside', textfont=dict(size=10, color='#6366f1')
            ))
            fig_inv_c.add_trace(go.Bar(
                name='Inversión (CLTV)',
                x=df_roi_cltv_seg['Segmento'], y=df_roi_cltv_seg['Inversion_€'],
                marker_color='#3b82f6',
                text=[f"{v/1000:.0f}K€" for v in df_roi_cltv_seg['Inversion_€']],
                textposition='outside', textfont=dict(size=10, color='white')
            ))
            fig_inv_c.add_trace(go.Bar(
                name='Beneficio Neto (CLTV)',
                x=df_roi_cltv_seg['Segmento'], y=df_roi_cltv_seg['Beneficio_Neto_€'],
                marker_color=[('#10b981' if v >= 0 else '#ef4444') for v in df_roi_cltv_seg['Beneficio_Neto_€']],
                text=[f"{v/1000:.0f}K€" for v in df_roi_cltv_seg['Beneficio_Neto_€']],
                textposition='outside', textfont=dict(size=10, color='white')
            ))
            fig_inv_c.update_layout(
                title="<b>Inversión vs Beneficio Neto</b>",
                xaxis_title="Segmento", yaxis_title="Euros (€)",
                barmode='group', template='plotly_dark', height=420,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(15,23,42,0.8)'),
                margin=dict(t=60, b=60, l=40, r=20)
            )
            st.plotly_chart(fig_inv_c, use_container_width=True)

        with cc2:
            seg_colors_c = ['#ef4444', '#f97316', '#f59e0b', '#84cc16', '#10b981']
            fig_funnel_c = go.Figure()
            fig_funnel_c.add_trace(go.Bar(
                name='Clientes Totales',
                x=df_roi_cltv_seg['Segmento'], y=df_roi_cltv_seg['N_Clientes'],
                marker_color='#334155',
                text=df_roi_cltv_seg['N_Clientes'], textposition='inside',
                textfont=dict(size=12, color='white')
            ))
            fig_funnel_c.add_trace(go.Bar(
                name='Captados (Solo Churn)',
                x=df_roi['Segmento'], y=df_roi['N_Captados'],
                marker_color=['rgba(239,68,68,0.45)', 'rgba(249,115,22,0.45)', 'rgba(245,158,11,0.45)', 'rgba(132,204,22,0.45)', 'rgba(16,185,129,0.45)'],
                text=df_roi['N_Captados'], textposition='inside',
                textfont=dict(size=11, color='white')
            ))
            fig_funnel_c.add_trace(go.Bar(
                name='Captados (CLTV)',
                x=df_roi_cltv_seg['Segmento'], y=df_roi_cltv_seg['N_Captados'],
                marker_color=seg_colors_c,
                text=df_roi_cltv_seg['N_Captados'], textposition='inside',
                textfont=dict(size=11, color='white')
            ))
            fig_funnel_c.update_layout(
                title="<b>Clientes Totales vs Captados</b>",
                xaxis_title="Segmento", yaxis_title="Nº Clientes",
                barmode='overlay', template='plotly_dark', height=420,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(15,23,42,0.8)'),
                margin=dict(t=60, b=60, l=40, r=20)
            )
            st.plotly_chart(fig_funnel_c, use_container_width=True)


    # ════════════════════════════════════════════════════════════════════════════
    # COMPARATIVA · se renderiza al final de ambos tabs
    # ════════════════════════════════════════════════════════════════════════════
    def _render_comparativa(key_suffix):
        st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🆚 Comparativa: Solo Churn vs Churn × CLTV")

        segmentos_ord  = ['MUY_ALTO', 'ALTO', 'MEDIO', 'BAJO', 'MUY_BAJO']
        roi_churn_vals = [df_roi.set_index('Segmento')['ROI'].get(s, 0) for s in segmentos_ord]
        roi_cltv_vals  = [roi_cltv_seg.get(s, 0) for s in segmentos_ord]

        c_left, c_right = st.columns(2)
        with c_left:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name='Solo Churn', x=segmentos_ord, y=roi_churn_vals,
                marker_color='#6366f1',
                text=[f"{v:.2f}x" for v in roi_churn_vals],
                textposition='outside', textfont=dict(size=11, color='white')
            ))
            fig_comp.add_trace(go.Bar(
                name='Churn × CLTV', x=segmentos_ord, y=roi_cltv_vals,
                marker_color='#3b82f6',
                text=[f"{v:.2f}x" for v in roi_cltv_vals],
                textposition='outside', textfont=dict(size=11, color='white')
            ))
            fig_comp.add_hline(y=1, line_dash='dash', line_color='#ef4444', line_width=1,
                               annotation_text='Punto equilibrio', annotation_position='right')
            fig_comp.update_layout(
                title="<b>ROI por Segmento: Solo Churn vs CLTV</b>",
                xaxis_title="Segmento", yaxis_title="ROI",
                barmode='group', template='plotly_dark', height=380,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(15,23,42,0.8)'),
                margin=dict(t=60, b=50, l=40, r=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True, key=f"comp_roi_{key_suffix}")

        with c_right:
            def _delta_html(v_new, v_old, fmt):
                diff = v_new - v_old
                color = '#10b981' if diff >= 0 else '#ef4444'
                sign  = '+' if diff >= 0 else ''
                return f'<span style="color:{color};font-size:0.82rem;font-weight:600">{sign}{fmt(diff)}</span>'

            rows_cmp = [
                ("Inversión",     f"{inv_global:,.0f} €",   f"{inv_global_c:,.0f} €",   _delta_html(inv_global_c,  inv_global,  lambda d: f"{d:,.0f} €")),
                ("Beneficio Neto",f"{bn_global:,.0f} €",    f"{bn_global_c:,.0f} €",    _delta_html(bn_global_c,   bn_global,   lambda d: f"{d:,.0f} €")),
                ("ROI Global",    f"{roi_global:.2f}x",     f"{roi_global_c:.2f}x",     _delta_html(roi_global_c,  roi_global,  lambda d: f"{d:.2f}x")),
                ("Captados",      f"{cap_global:,}",        f"{cap_global_c:,}",        _delta_html(cap_global_c,  cap_global,  lambda d: f"{int(d):,}")),
            ]
            filas_html = "".join(f"""
                <tr>
                    <td style="color:#94a3b8;font-weight:600">{r[0]}</td>
                    <td style="color:#a5b4fc">{r[1]}</td>
                    <td style="color:#93c5fd">{r[2]}</td>
                    <td>{r[3]}</td>
                </tr>""" for r in rows_cmp)
            st.markdown(f"""
            <div style="margin-top:1rem">
            <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:8px;font-weight:600">MÉTRICAS GLOBALES COMPARADAS</p>
            <table class="econ-table">
                <thead><tr>
                    <th>Métrica</th>
                    <th style="color:#a5b4fc">Solo Churn</th>
                    <th style="color:#93c5fd">Churn × CLTV</th>
                    <th>Diferencia</th>
                </tr></thead>
                <tbody>{filas_html}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

    with tab1:
        _render_comparativa("tab1")

    with tab2:
        _render_comparativa("tab2")
