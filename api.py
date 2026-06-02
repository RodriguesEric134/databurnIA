import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Inicialização da API
app = FastAPI(
    title="DataBurn Monitor API",
    description="API REST para predição e monitoramento inteligente de risco de queimadas. Projetada para integração com RPA e sistemas de resposta.",
    version="1.0.0"
)

# Permitir CORS (para acesso de diferentes clientes Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/best_model.pkl"
model = None

# Carregar o modelo treinado na inicialização
def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Modelo treinado não encontrado em '{MODEL_PATH}'. "
                "Execute o pipeline de treinamento primeiro usando 'python src/train_pipeline.py'."
            )
        model = joblib.load(MODEL_PATH)
    return model

# Modelo Pydantic para validação das requisições de entrada (14 colunas do dataset)
class FirePredictionInput(BaseModel):
    latitude: float = Field(..., description="Latitude do local", example=-15.7801)
    longitude: float = Field(..., description="Longitude do local", example=-47.9292)
    temperature_2m: float = Field(..., description="Temperatura do ar a 2 metros de altura em °C", example=33.5)
    relative_humidity_2m: float = Field(..., description="Umidade relativa do ar a 2 metros em %", example=32.0)
    precipitation: float = Field(..., description="Precipitação acumulada em mm", example=0.0)
    wind_speed_10m: float = Field(..., description="Velocidade do vento a 10 metros em km/h", example=18.5)
    soil_moisture_index: float = Field(..., description="Índice de umidade do solo (0.0 a 1.0)", example=0.22)
    ndvi: float = Field(..., description="Índice de Vegetação por Diferença Normalizada (0.0 a 1.0)", example=0.42)
    days_without_rain: int = Field(..., description="Quantidade de dias consecutivos sem chuva", example=25)
    brightness: float = Field(..., description="Temperatura de brilho medida por satélite em Kelvin", example=332.5)
    frp: float = Field(..., description="Poder Radiativo do Fogo (FRP) em MW", example=45.0)
    confidence: float = Field(..., description="Nível de confiança da detecção em % (0 a 100)", example=85.0)
    month: int = Field(..., description="Mês do ano (1 a 12)", example=8)
    biome: str = Field(..., description="Nome do bioma brasileiro", example="Cerrado")

    class Config:
        schema_extra = {
            "example": {
                "latitude": -15.7801,
                "longitude": -47.9292,
                "temperature_2m": 33.5,
                "relative_humidity_2m": 32.0,
                "precipitation": 0.0,
                "wind_speed_10m": 18.5,
                "soil_moisture_index": 0.22,
                "ndvi": 0.42,
                "days_without_rain": 25,
                "brightness": 332.5,
                "frp": 45.0,
                "confidence": 85.0,
                "month": 8,
                "biome": "Cerrado"
            }
        }

# Resposta de Predição formatada conforme especificações
class FirePredictionResponse(BaseModel):
    risk_class: str = Field(..., description="Classe do risco de queimada (baixo, medio, alto, critico)")
    risk_score: float = Field(..., description="Score probabilístico estimado para risco de queimada (0.0 a 1.0)")
    risk_probability: str = Field(..., description="Probabilidade estimada formatada em porcentagem")
    alert_required: bool = Field(..., description="Flag indicativa se o alerta de combate ou resposta deve ser emitido (True para alto/crítico)")
    model_version: str = Field(..., description="Versão do modelo preditivo utilizado")


@app.get("/")
def read_root():
    """Retorna metadados gerais sobre o serviço DataBurn Monitor."""
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "project": "DataBurn Monitor",
        "description": "Plataforma Inteligente para Monitoramento e Previsão de Risco de Queimadas",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "status": "online" if model_loaded else "waiting_for_model",
        "institution": "FIAP - Engenharia de Software (Global Solution 2026)"
    }


@app.post("/predict", response_model=FirePredictionResponse)
def predict_fire_risk(payload: FirePredictionInput):
    """
    Recebe os parâmetros climáticos e espaciais de um local, avalia o risco
    usando o modelo preditivo de Machine Learning treinado e retorna a
    classificação e probabilidade de risco detalhadas, ideias para automação RPA.
    """
    try:
        # Tenta carregar o modelo
        predictor = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Cria dataframe a partir do payload recebido (com as mesmas colunas que o modelo espera)
    input_data = pd.DataFrame([{
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "temperature_2m": payload.temperature_2m,
        "relative_humidity_2m": payload.relative_humidity_2m,
        "precipitation": payload.precipitation,
        "wind_speed_10m": payload.wind_speed_10m,
        "soil_moisture_index": payload.soil_moisture_index,
        "ndvi": payload.ndvi,
        "days_without_rain": payload.days_without_rain,
        "brightness": payload.brightness,
        "frp": payload.frp,
        "confidence": payload.confidence,
        "month": payload.month,
        "biome": payload.biome
    }])

    # Engenharia de Atributos Dinâmica (Feature Engineering exigida pelo checklist)
    input_data["drought_index"] = input_data["temperature_2m"] * (100.0 - input_data["relative_humidity_2m"]) / 100.0
    input_data["vegetation_dryness"] = input_data["days_without_rain"] * (1.0 - input_data["ndvi"])

    try:
        # Obter probabilidade da classe 1 (alto risco de queimada)
        if hasattr(predictor, "predict_proba"):
            score = float(predictor.predict_proba(input_data)[0][1])
        else:
            # Caso o modelo não tenha predict_proba (fallback improvável)
            pred = predictor.predict(input_data)[0]
            score = 1.0 if pred == 1 else 0.0

        # Classificação em faixas de risco solicitada:
        # baixo: < 0.40
        # medio: 0.40 a 0.64
        # alto: 0.65 a 0.84
        # critico: >= 0.85
        if score < 0.40:
            risk_class = "baixo"
            alert_required = False
        elif score < 0.65:
            risk_class = "medio"
            alert_required = False
        elif score < 0.85:
            risk_class = "alto"
            alert_required = True
        else:
            risk_class = "critico"
            alert_required = True

        return FirePredictionResponse(
            risk_class=risk_class,
            risk_score=round(score, 4),
            risk_probability=f"{score:.2%}",
            alert_required=alert_required,
            model_version="1.0.0"
        )

    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno no processamento ou inferência do modelo: {str(err)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Executa a API na porta local 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
