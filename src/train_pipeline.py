import os
import joblib
import shap
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42

DATA_PATH = "data/spacefire_dataset.csv"
MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
SHAP_OUTPUT_PATH = "reports/shap_summary.png"


def generate_synthetic_dataset(n_rows: int = 1200) -> pd.DataFrame:
    """
    Gera dataset sintético coerente com o problema do SpaceFire Monitor.
    Deve ser substituído futuramente por integração NASA FIRMS + Open-Meteo.
    """

    np.random.seed(RANDOM_STATE)

    biomes = ["Amazonia", "Cerrado", "Mata Atlantica", "Caatinga", "Pantanal", "Pampa"]

    df = pd.DataFrame({
        "latitude": np.random.uniform(-33.0, 5.0, n_rows),
        "longitude": np.random.uniform(-73.0, -34.0, n_rows),
        "temperature_2m": np.random.normal(29, 5, n_rows),
        "relative_humidity_2m": np.random.normal(55, 18, n_rows),
        "precipitation": np.random.exponential(2.5, n_rows),
        "wind_speed_10m": np.random.normal(12, 4, n_rows),
        "soil_moisture_index": np.random.uniform(0, 1, n_rows),
        "ndvi": np.random.uniform(0.1, 0.9, n_rows),
        "days_without_rain": np.random.randint(0, 45, n_rows),
        "brightness": np.random.normal(315, 18, n_rows),
        "frp": np.random.exponential(20, n_rows),
        "confidence": np.random.uniform(20, 100, n_rows),
        "month": np.random.randint(1, 13, n_rows),
        "biome": np.random.choice(biomes, n_rows)
    })

    # Limpeza de limites realistas
    df["relative_humidity_2m"] = df["relative_humidity_2m"].clip(5, 100)
    df["temperature_2m"] = df["temperature_2m"].clip(5, 48)
    df["wind_speed_10m"] = df["wind_speed_10m"].clip(0, 35)
    df["brightness"] = df["brightness"].clip(280, 390)

    # Regra sintética para target: risco cresce com calor, seca, vento, brilho e FRP
    risk_score = (
        0.25 * (df["temperature_2m"] > 32).astype(int) +
        0.20 * (df["relative_humidity_2m"] < 40).astype(int) +
        0.15 * (df["days_without_rain"] > 15).astype(int) +
        0.15 * (df["wind_speed_10m"] > 15).astype(int) +
        0.15 * (df["brightness"] > 325).astype(int) +
        0.10 * (df["frp"] > 25).astype(int) +
        0.10 * (df["soil_moisture_index"] < 0.35).astype(int)
    )

    noise = np.random.normal(0, 0.12, n_rows)
    df["fire_risk"] = ((risk_score + noise) >= 0.45).astype(int)

    return df


def load_or_create_dataset() -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)

    regenerate = False
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            if len(df) < 1000 or len(df.columns) < 10:
                print(f"Dataset em {DATA_PATH} não atende aos requisitos (linhas: {len(df)}/1000, colunas: {len(df.columns)}/10). Regenerando...")
                regenerate = True
        except Exception as e:
            print(f"Erro ao ler dataset existente ({e}). Regenerando...")
            regenerate = True
    else:
        regenerate = True

    if regenerate:
        df = generate_synthetic_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Novo dataset sintético gerado com {len(df)} linhas e {len(df.columns)} colunas em: {DATA_PATH}")

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc
    }

    print(f"\n===== {name} =====")
    print(pd.DataFrame([metrics]))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics, model


def run_shap_analysis(best_pipeline, X_train, X_test):
    """
    Gera interpretabilidade usando SHAP.
    Para modelos baseados em árvore, TreeExplainer costuma ser mais eficiente.
    """

    os.makedirs("reports", exist_ok=True)

    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    X_test_transformed_df = pd.DataFrame(
        X_test_transformed,
        columns=feature_names
    )

    try:
        # Tenta usar o explainer genérico do SHAP
        explainer = shap.Explainer(model, X_test_transformed_df)
        shap_values = explainer(X_test_transformed_df)
    except Exception as e:
        print(f"Aviso: Erro ao instanciar Explainer genérico ({e}). Tentando TreeExplainer ou LinearExplainer...")
        try:
            if "LogisticRegression" in str(type(model)):
                explainer = shap.LinearExplainer(model, X_test_transformed_df)
                shap_values = explainer.shap_values(X_test_transformed_df)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_transformed_df)
        except Exception as e2:
            print(f"Erro ao instanciar Explainer alternativo ({e2}). Usando KernelExplainer com amostra pequena...")
            # Fallback para KernelExplainer com uma amostra menor para evitar travamento
            background = X_test_transformed_df.sample(min(50, len(X_test_transformed_df)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test_transformed_df.sample(min(100, len(X_test_transformed_df)), random_state=42))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    try:
        # Se shap_values for uma lista (comum em TreeExplainer clássico no sklearn com multiclass)
        if isinstance(shap_values, list):
            # Usar shap values da classe de alto risco (1)
            shap.summary_plot(shap_values[1], X_test_transformed_df, show=False)
        elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
            # Para novas APIs do SHAP que retornam arrays 3D (samples, features, classes)
            shap.summary_plot(shap_values.values[:, :, 1], X_test_transformed_df, show=False)
        else:
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
    except Exception as e_plot:
        print(f"Aviso: Erro ao plotar SHAP customizado ({e_plot}). Tentando plotar objeto bruto...")
        try:
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
        except Exception as e_plot_final:
            print(f"Erro crítico ao plotar SHAP: {e_plot_final}")
            # Salva uma imagem em branco informativa em caso de falha catastrófica para evitar quebrar o pipeline
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "Gráfico SHAP Indisponível\n(Erro de compatibilidade da biblioteca)", ha='center', va='center')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(SHAP_OUTPUT_PATH, bbox_inches="tight")
    plt.close()

    print(f"\nSHAP summary salvo em: {SHAP_OUTPUT_PATH}")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_or_create_dataset()

    target = "fire_risk"

    X = df.drop(columns=[target])
    y = df[target]

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    results = []
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        metrics, trained_pipeline = evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(metrics)
        trained_pipelines[name] = trained_pipeline

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score",
        ascending=False
    )

    print("\n===== Comparação Final =====")
    print(results_df)

    # Salva o arquivo de métricas em reports/model_metrics.csv
    metrics_csv_path = "reports/model_metrics.csv"
    results_df.to_csv(metrics_csv_path, index=False)
    print(f"\nMétricas dos modelos salvas em: {metrics_csv_path}")

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]

    joblib.dump(best_pipeline, MODEL_PATH)

    print(f"\nMelhor modelo: {best_model_name}")
    print(f"Modelo salvo em: {MODEL_PATH}")

    run_shap_analysis(best_pipeline, X_train, X_test)


if __name__ == "__main__":
    main()