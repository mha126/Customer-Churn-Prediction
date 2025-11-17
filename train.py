import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def build_pipeline(categorical_cols, numeric_cols):
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_cols),
            ("num", num_transformer, numeric_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )
    return pipe

def main():
    data_path = "data/customer_churn.csv"
    df = load_data(data_path)

    target_col = "churn"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    categorical_cols = ["contract_type", "payment_method"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline(categorical_cols, numeric_cols)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)

    # 🔥 Added accuracy metric
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = "churn_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved trained model to {model_path}")

if __name__ == "__main__":
    main()
