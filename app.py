import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "reports/model_metrics.csv"
SHAP_PATH = "reports/shap_summary.png"


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def main():
    st.set_page_config(
        page_title="DataBurn Monitor - Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar - Tema e Informações Gerais
    with st.sidebar:
        st.markdown("### Configurações")
        theme_choice = st.selectbox(
            "Tema do Dashboard",
            ["Escuro", "Claro"],
            index=0,
            help="Alterne entre o tema escuro ou claro para a interface."
        )

        st.markdown("---")
        st.markdown("### DataBurn Monitor")
        st.markdown(
            """
            Plataforma para detecção precoce, previsão probabilística e resposta a queimadas florestais no Brasil.
            
            **Global Solution 2026**  
            *FIAP - Engenharia de Software*
            
            * **Disciplina**: Generative AI for Engineering
            * **Versão do Modelo**: 1.0.0
            """
        )
        
        model = load_model()
        if model is not None:
            st.success("Modelo de ML Ativo")
        else:
            st.error("Aguardando Treinamento")
            st.info("Execute 'python src/train_pipeline.py' no terminal para treinar e salvar o melhor modelo.")

        st.markdown("---")
        st.markdown("### Variáveis Mapeadas")
        st.caption("Clima: Open-Meteo (Temperatura, Umidade, Chuva)")
        st.caption("Satélites: NASA FIRMS (FRP, Confiança, Brilho)")
        st.caption("Vegetação: INPE e Copernicus (NDVI, Bioma)")

    # Aplicação do CSS Customizado de acordo com o Tema Escolhido (Estilo Minimalista e Leitura Perfeita)
    if theme_choice == "Escuro":
        bg_color = "#0b0f19"
        text_color = "#f1f5f9"
        card_bg = "#1e293b"
        card_border = "#334155"
        card_text = "#f8fafc"
        label_color = "#94a3b8"
        btn_bg = "#f8fafc"
        btn_text = "#0f172a"
        btn_hover_bg = "#e2e8f0"
        
        st.markdown(
            f"""
            <style>
            /* Forçar cores de texto claras no tema escuro para todos os elementos padrão do Streamlit */
            .stApp, 
            .stApp p, 
            .stApp label, 
            .stApp span, 
            .stApp h1, 
            .stApp h2, 
            .stApp h3, 
            .stApp h4, 
            .stApp h5, 
            .stApp h6, 
            .stApp li,
            div[data-testid="stWidgetLabel"] p,
            div[data-testid="stSliderValue"] p,
            div[data-testid="stMarkdownContainer"] p,
            button[data-baseweb="tab"] span,
            button[data-baseweb="tab"] p,
            .stSelectbox div[role="button"] {{
                color: {text_color} !important;
            }}
            
            .stApp {{
                background-color: {bg_color} !important;
            }}
            section[data-testid="stSidebar"] {{
                background-color: #0f172a !important;
                border-right: 1px solid #1e293b !important;
            }}
            .stButton>button {{
                background-color: {btn_bg} !important;
                color: {btn_text} !important;
                border: 1px solid {card_border} !important;
                border-radius: 4px !important;
                padding: 10px 24px !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
                width: 100% !important;
            }}
            .stButton>button:hover {{
                background-color: {btn_hover_bg} !important;
                color: {btn_text} !important;
                border-color: #cbd5e1 !important;
            }}
            .metric-card {{
                background-color: {card_bg} !important;
                border: 1px solid {card_border} !important;
                border-radius: 6px !important;
                padding: 15px !important;
                text-align: center !important;
            }}
            .metric-value {{
                font-size: 24px !important;
                font-weight: 600 !important;
            }}
            .metric-label {{
                font-size: 12px !important;
                color: {label_color} !important;
                margin-bottom: 5px !important;
            }}
            div[data-testid="stForm"] {{
                background-color: #111827 !important;
                border: 1px solid #1f2937 !important;
                border-radius: 8px !important;
                padding: 20px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:  # Tema Claro
        bg_color = "#ffffff"
        text_color = "#0f172a"
        card_bg = "#f1f5f9"
        card_border = "#e2e8f0"
        card_text = "#0f172a"
        label_color = "#64748b"
        btn_bg = "#0f172a"
        btn_text = "#ffffff"
        btn_hover_bg = "#1e293b"
        
        st.markdown(
            f"""
            <style>
            /* Forçar cores de texto escuras no tema claro para todos os elementos padrão do Streamlit */
            .stApp, 
            .stApp p, 
            .stApp label, 
            .stApp span, 
            .stApp h1, 
            .stApp h2, 
            .stApp h3, 
            .stApp h4, 
            .stApp h5, 
            .stApp h6, 
            .stApp li,
            div[data-testid="stWidgetLabel"] p,
            div[data-testid="stSliderValue"] p,
            div[data-testid="stMarkdownContainer"] p,
            button[data-baseweb="tab"] span,
            button[data-baseweb="tab"] p,
            .stSelectbox div[role="button"] {{
                color: {text_color} !important;
            }}
            
            .stApp {{
                background-color: {bg_color} !important;
            }}
            section[data-testid="stSidebar"] {{
                background-color: #f8fafc !important;
                border-right: 1px solid #e2e8f0 !important;
            }}
            .stButton>button {{
                background-color: {btn_bg} !important;
                color: {btn_text} !important;
                border: 1px solid {btn_bg} !important;
                border-radius: 4px !important;
                padding: 10px 24px !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
                width: 100% !important;
            }}
            .stButton>button:hover {{
                background-color: {btn_hover_bg} !important;
                color: {btn_text} !important;
                border-color: {btn_hover_bg} !important;
            }}
            .metric-card {{
                background-color: {card_bg} !important;
                border: 1px solid {card_border} !important;
                border-radius: 6px !important;
                padding: 15px !important;
                text-align: center !important;
            }}
            .metric-value {{
                font-size: 24px !important;
                font-weight: 600 !important;
            }}
            .metric-label {{
                font-size: 12px !important;
                color: {label_color} !important;
                margin-bottom: 5px !important;
            }}
            div[data-testid="stForm"] {{
                background-color: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 8px !important;
                padding: 20px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    # Título Principal (Sem Emojis, com Acentuação Correta)
    st.markdown(
        """
        <div style="display:flex; align-items:center; margin-bottom: 25px;">
            <h1 style="margin:0; font-weight:600; font-size:2rem; letter-spacing:-0.5px;">DataBurn Monitor</h1>
            <span style="background-color:#0f172a; color:#f8fafc; border: 1px solid #334155; padding:4px 10px; border-radius:15px; font-size:11px; font-weight:500; margin-left:15px; vertical-align:middle;">RELEASE ACADÊMICO</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Abas da Interface (Sem Emojis)
    tab1, tab2, tab3 = st.tabs([
        "Calculadora de Risco", 
        "Comparativo de Modelos", 
        "Explicabilidade SHAP"
    ])

    # --- ABA 1: CALCULADORA DE RISCO ---
    with tab1:
        st.markdown("Insira as condições climáticas e espaciais observadas para calcular o risco estimado:")
        
        with st.form("risk_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Localização e Clima")
                latitude = st.number_input("Latitude", value=-15.0, min_value=-34.0, max_value=6.0, step=0.1, help="Coordenada geográfica vertical.")
                longitude = st.number_input("Longitude", value=-47.0, min_value=-74.0, max_value=-34.0, step=0.1, help="Coordenada geográfica horizontal.")
                temperature_2m = st.slider("Temperatura (°C)", 0.0, 50.0, 32.0, help="Temperatura do ar a 2 metros de altura.")
                relative_humidity_2m = st.slider("Umidade Relativa (%)", 0.0, 100.0, 35.0, help="Umidade relativa do ar (valores baixos aumentam o risco).")
                precipitation = st.slider("Precipitação (mm)", 0.0, 50.0, 0.5, step=0.1, help="Volume de chuva recente acumulada em mm.")
                wind_speed_10m = st.slider("Velocidade do Vento (km/h)", 0.0, 50.0, 18.0, help="Velocidade do vento medida a 10 metros de altura.")
                soil_moisture_index = st.slider("Índice de Umidade do Solo", 0.0, 1.0, 0.25, step=0.01, help="Umidade do solo superficial (0 = seco, 1 = saturado).")

            with col2:
                st.markdown("##### Sensores Espaciais e Vegetação")
                ndvi = st.slider("Índice de Vegetação (NDVI)", 0.0, 1.0, 0.45, step=0.01, help="NDVI (valores baixos indicam vegetação seca ou escassa).")
                days_without_rain = st.slider("Dias Sem Chuva", 0, 60, 20, help="Quantidade de dias consecutivos sem precipitação.")
                brightness = st.slider("Brilho Térmico (Kelvin)", 280.0, 390.0, 330.0, help="Temperatura de brilho medida pelos sensores MODIS/VIIRS do satélite.")
                frp = st.slider("Poder Radiativo do Fogo (FRP)", 0.0, 200.0, 35.0, help="FRP em Megawatts. Intensidade da liberação de energia térmica.")
                confidence = st.slider("Confiança do Satélite (%)", 0.0, 100.0, 80.0, help="Porcentagem de certeza do satélite sobre a anomalia térmica.")
                month = st.slider("Mês do Ano", 1, 12, 8, help="Mês observado (queimadas concentram-se no segundo semestre).")
                biome = st.selectbox(
                    "Bioma de Referência",
                    ["Amazonia", "Cerrado", "Mata Atlantica", "Caatinga", "Pantanal", "Pampa"]
                )

            # Botão de Envio (Sem Emojis)
            submit_button = st.form_submit_button("CALCULAR RISCO DE QUEIMADA")

        # Processamento e exibição de resultados ao submeter
        if submit_button:
            if model is None:
                st.error("Erro: O modelo preditivo ainda não foi treinado. Por favor, execute o script 'train_pipeline.py' na pasta src primeiro.")
            else:
                # DataFrame com as colunas estruturadas exatamente como no treinamento
                input_df = pd.DataFrame([{
                    "latitude": latitude,
                    "longitude": longitude,
                    "temperature_2m": temperature_2m,
                    "relative_humidity_2m": relative_humidity_2m,
                    "precipitation": precipitation,
                    "wind_speed_10m": wind_speed_10m,
                    "soil_moisture_index": soil_moisture_index,
                    "ndvi": ndvi,
                    "days_without_rain": days_without_rain,
                    "brightness": brightness,
                    "frp": frp,
                    "confidence": confidence,
                    "month": month,
                    "biome": biome
                }])

                # Engenharia de Atributos Dinâmica (Feature Engineering exigida pelo checklist)
                input_df["drought_index"] = input_df["temperature_2m"] * (100.0 - input_df["relative_humidity_2m"]) / 100.0
                input_df["vegetation_dryness"] = input_df["days_without_rain"] * (1.0 - input_df["ndvi"])

                # Cálculo de Predição e Score
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                # Lógica de Classificação em Faixas sem gaps decimais e concordância perfeita
                if probability < 0.40:
                    risk_class = "BAIXO"
                    alert_border_color = "#48bb78"  # Verde
                    alert_msg = "**Risco Baixo**: Condições ambientais normais e seguras. Nenhuma ação imediata é necessária."
                    alert_class = "baixo"
                    alert_required = False
                elif probability < 0.65:
                    risk_class = "MÉDIO"
                    alert_border_color = "#ecc94b"  # Amarelo
                    alert_msg = "**Risco Médio**: Condições favoráveis à propagação de fogo sob ignição artificial. Recomenda-se vigilância ativa."
                    alert_class = "medio"
                    alert_required = False
                elif probability < 0.85:
                    risk_class = "ALTO"
                    alert_border_color = "#ed8936"  # Laranja
                    alert_msg = "**Risco Alto**: Atenção! Clima seco, quente e vegetação propícia. Disparando alerta de RPA automatizado para brigadas locais."
                    alert_class = "alto"
                    alert_required = True
                else:
                    risk_class = "CRÍTICO"
                    alert_border_color = "#e53e3e"  # Vermelho
                    alert_msg = "**Risco Crítico (Alerta Máximo)**: Condições de extrema inflamabilidade. Evacuações preventivas e ações de combate de urgência exigidas! Disparo de alerta de RPA crítico."
                    alert_class = "critico"
                    alert_required = True

                # Painel de Resposta Visual do Risco (Sem Emojis)
                st.markdown("### Resultado da Análise de Risco")
                
                col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
                
                with col_res1:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 5px solid {alert_border_color} !important;">
                            <div class="metric-label" style="color: {label_color} !important;">CLASSIFICAÇÃO DE RISCO</div>
                            <div class="metric-value" style="color: {alert_border_color} !important;">{risk_class}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col_res2:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 5px solid #4299e1 !important;">
                            <div class="metric-label" style="color: {label_color} !important;">PROBABILIDADE ESTIMADA</div>
                            <div class="metric-value" style="color: #4299e1 !important;">{probability:.2%}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                with col_res3:
                    alert_text_color = "#48bb78" if not alert_required else "#f56565"
                    alert_status_label = "Dispensado" if not alert_required else "Exigido"
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 5px solid {alert_text_color} !important;">
                            <div class="metric-label" style="color: {label_color} !important;">DISPARO DE ALERTA (RPA)</div>
                            <div class="metric-value" style="color: {alert_text_color} !important;">{alert_status_label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Caixa de aviso dinâmico minimalista baseada no nível de risco (Fonte e fundos perfeitamente harmonizados)
                st.markdown(
                    f"""
                    <div style="background-color: {card_bg}; border: 1px solid {alert_border_color}; border-left: 5px solid {alert_border_color}; padding: 15px; border-radius: 4px; margin-top: 15px; color: {text_color} !important;">
                        {alert_msg}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Detalhes do JSON enviado (Sem Emojis, com Acentuação Correta)
                with st.expander("Ver Payload JSON (RPA / API Integration)"):
                    json_payload = {
                        "risk_class": alert_class,
                        "risk_score": round(probability, 4),
                        "risk_probability": f"{probability:.2%}",
                        "alert_required": alert_required,
                        "model_version": "1.0.0"
                    }
                    st.json(json_payload)

    # --- ABA 2: COMPARATIVO DE MODELOS ---
    with tab2:
        st.markdown("### Benchmarking dos Modelos Avaliados")
        st.markdown(
            """
            Para esta entrega acadêmica, o pipeline treina de forma automatizada três algoritmos clássicos de aprendizado de máquina supervisionado para comparação:
            1. **Regressão Logística**: Modelo linear estável como baseline.
            2. **Random Forest**: Algoritmo de floresta de decisão, que lida com interações não lineares de forma robusta.
            3. **Gradient Boosting (GBDT)**: Classificador baseado em boosting de árvores de decisão sequenciais.
            """
        )

        if os.path.exists(METRICS_PATH):
            df_metrics = pd.read_csv(METRICS_PATH)
            
            # Tabela limpa formatada
            st.dataframe(
                df_metrics.style.format({
                    "accuracy": "{:.2%}",
                    "precision": "{:.2%}",
                    "recall": "{:.2%}",
                    "f1_score": "{:.2%}",
                    "roc_auc": "{:.2%}"
                }),
                use_container_width=True
            )
            
            # Gráfico de comparação minimalista
            st.markdown("##### Comparação Visual de Performance (F1-Score)")
            st.bar_chart(data=df_metrics, x="model", y="f1_score", color="#3182ce")
        else:
            st.warning("Nenhuma métrica encontrada em 'reports/model_metrics.csv'. Por favor, execute o script de treinamento primeiro para gerar o benchmarking.")

    # --- ABA 3: EXPLICABILIDADE SHAP ---
    with tab3:
        st.markdown("### Interpretabilidade Científica com SHAP (SHapley Additive exPlanations)")
        st.markdown(
            """
            Modelos de IA modernos não devem atuar como 'caixas-pretas'. A fim de prover rigor para o desenvolvimento de software espacial, integramos o **SHAP**, um framework matemático fundamentado na **Teoria dos Jogos Cooperativos**.
            
            O SHAP calcula a contribuição marginal de cada variável climática e de satélite para deslocar a probabilidade de risco de queimadas para cima ou para baixo.
            """
        )

        col_shap_text, col_shap_img = st.columns([1, 2])

        with col_shap_text:
            st.markdown(
                """
                **Como interpretar a explicabilidade:**
                
                * **Eixo Y**: Variáveis em ordem decrescente de importância global. As variáveis no topo são as que mais impactam as predições do modelo.
                * **Eixo X (SHAP Value)**: Força do impacto na predição. Valores à direita (positivos) empurram a predição para 'Risco de Queimada = Alto', enquanto valores à esquerda (negativos) empurram para 'Risco de Queimada = Baixo'.
                * **Gradiente de Cor**: Representa o valor real observado da variável naquele ponto. Vermelho indica valor alto (ex: temperatura alta) e azul indica valor baixo (ex: baixa umidade relativa).
                
                *Exemplo*: Pontos vermelhos localizados à direita na variável de temperatura mostram que alta temperatura desloca fortemente o risco para cima.
                """
            )

        with col_shap_img:
            if os.path.exists(SHAP_PATH):
                st.image(SHAP_PATH, caption="SHAP Summary Plot - Explicabilidade do Melhor Modelo Treinado", use_container_width=True)
            else:
                st.info("Gráfico do SHAP indisponível em 'reports/shap_summary.png'. Ele é gerado de forma automatizada ao final do pipeline de treinamento.")


if __name__ == "__main__":
    main()