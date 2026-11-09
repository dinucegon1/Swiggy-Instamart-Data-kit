"""Customer segmentation (K-Means + PCA) and lapse-classification models.
Ensures leakage-free features with strict temporal train/observation splits.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.dummy import DummyClassifier

DB_PATH = "data/instamart.db"
MODEL_DIR = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data_from_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run data_pipeline.py first.")
    conn = sqlite3.connect(DB_PATH)
    orders = pd.read_sql("SELECT * FROM orders WHERE status = 'completed'", conn)
    conn.close()
    return orders

def compute_rfm(orders_df, snapshot_date="2025-09-01"):
    df = orders_df.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"])
    snap = pd.to_datetime(snapshot_date)
    
    # Filter strictly before snapshot date to prevent leakage for churn/lapse
    train_orders = df[df["order_ts"] < snap]
    
    rfm = train_orders.groupby("customer_id").agg(
        recency=("order_ts", lambda x: (snap - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    
    return rfm, train_orders

def rfm_and_segments(n_clusters=4):
    orders = load_data_from_db()
    rfm, train_orders = compute_rfm(orders, "2025-09-01")
    
    if rfm.empty:
        raise ValueError("RFM dataframe is empty. Check orders data.")
        
    X_rfm = rfm[["recency", "frequency", "monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rfm)
    
    # K-Means clustering diagnostics / execution
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)
    
    # PCA for dimensional visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    rfm["pca_1"] = pca_coords[:, 0]
    rfm["pca_2"] = pca_coords[:, 1]
    
    # Diagnostics summary across clusters
    diagnostics = rfm.groupby("cluster").agg(
        count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()
    
    # Save model and artifacts
    joblib.dump({"scaler": scaler, "kmeans": kmeans, "pca": pca}, os.path.join(MODEL_DIR, "segmentation_pipeline.pkl"))
    
    return rfm, diagnostics, diagnostics, pca.explained_variance_ratio_

def build_lapse_model():
    orders = load_data_from_db()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    
    # Features stop before the August-September observation window.
    snapshot = pd.Timestamp("2025-08-01")
    feat_orders = orders[orders["order_ts"] < pd.to_datetime("2025-08-01")]
    
    rfm_feat = feat_orders.groupby("customer_id").agg(
        recency=("order_ts", lambda x: (snapshot - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    
    # August-September orders are used only to assign the target.
    obs_orders = orders[(orders["order_ts"] >= pd.to_datetime("2025-08-01")) & (orders["order_ts"] < pd.to_datetime("2025-10-01"))]
    active_in_obs = set(obs_orders["customer_id"].unique())
    
    rfm_feat["churn"] = (~rfm_feat["customer_id"].isin(active_in_obs)).astype(int)

    conn = sqlite3.connect(DB_PATH)
    customers = pd.read_sql_query(
        "SELECT customer_id, city, age, gender, signup_date FROM customers", conn
    )
    conn.close()
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[~customers["age"].between(18, 100), "age"] = np.nan
    signup_date = pd.to_datetime(customers["signup_date"], errors="coerce")
    customers["signup_tenure_days"] = (snapshot - signup_date).dt.days
    customers.loc[customers["signup_tenure_days"] < 0, "signup_tenure_days"] = np.nan
    rfm_feat = rfm_feat.merge(
        customers[["customer_id", "city", "gender", "age", "signup_tenure_days"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    rfm_feat["recency_x_frequency"] = rfm_feat["recency"] * rfm_feat["frequency"]
    rfm_feat["frequency_x_monetary"] = rfm_feat["frequency"] * rfm_feat["monetary"]
    rfm_feat["average_order_value"] = rfm_feat["monetary"] / rfm_feat["frequency"]
    
    # Save processed customer features dataset
    os.makedirs("data", exist_ok=True)
    rfm_feat.to_csv("data/customer_features.csv", index=False)
    
    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "age",
        "signup_tenure_days",
        "recency_x_frequency",
        "frequency_x_monetary",
        "average_order_value",
    ]
    categorical_features = ["city", "gender"]
    X = rfm_feat[numeric_features + categorical_features]
    y = rfm_feat["churn"]
    
    if y.nunique() < 2 or y.value_counts().min() < 4:
        raise ValueError(
            "Churn labels need at least four examples of each class for the "
            "stratified holdout and 3-fold cross-validation."
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def make_preprocessor(scale_numeric):
        numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(numeric_steps)
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ])

    def make_pipeline(classifier, scale_numeric, select_features):
        steps = [("preprocessor", make_preprocessor(scale_numeric))]
        if select_features:
            steps.append((
                "feature_selection",
                SelectFromModel(
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=42,
                    ),
                    threshold="median",
                ),
            ))
        steps.append(("model", classifier))
        return Pipeline(steps)

    def metrics(y_true, probabilities, threshold=0.5):
        predictions = (probabilities >= threshold).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
        }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    baseline_estimators = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, random_state=42),
            False,
        ),
    }
    baseline_metrics = {}
    for name, (estimator, scale_numeric) in baseline_estimators.items():
        baseline = make_pipeline(estimator, scale_numeric, select_features=False)
        baseline.fit(X_train, y_train)
        baseline_metrics[name] = metrics(
            y_test, baseline.predict_proba(X_test)[:, 1]
        )

    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_metrics["majority_class"] = metrics(
        y_test, majority.predict_proba(X_test)[:, 1]
    )

    searches = {
        "logistic_regression": GridSearchCV(
            make_pipeline(
                LogisticRegression(max_iter=2000, random_state=42),
                scale_numeric=True,
                select_features=True,
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        ),
        "random_forest": GridSearchCV(
            make_pipeline(
                RandomForestClassifier(random_state=42),
                scale_numeric=False,
                select_features=True,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3],
                "model__class_weight": [None, "balanced"],
            },
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        ),
    }
    for search in searches.values():
        search.fit(X_train, y_train)

    best_name, best_search = max(
        searches.items(), key=lambda item: item[1].best_score_
    )
    best_model = best_search.best_estimator_
    model_search_results = {
        name: {
            "cv_f1": float(search.best_score_),
            "best_params": search.best_params_,
        }
        for name, search in searches.items()
    }

    # Select the F1 threshold from out-of-fold training probabilities, never the test set.
    threshold_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
    train_probabilities = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=threshold_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_train, train_probabilities, pos_label=1
    )
    if len(thresholds):
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        )
        decision_threshold = float(thresholds[int(np.argmax(f1_values))])
    else:
        decision_threshold = 0.5

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    default_metrics = metrics(y_test, test_probabilities)
    tuned_metrics = metrics(y_test, test_probabilities, decision_threshold)
    preprocessor = best_model.named_steps["preprocessor"]
    transformed_features = preprocessor.get_feature_names_out()
    selected_features = transformed_features[
        best_model.named_steps["feature_selection"].get_support()
    ].tolist()

    joblib.dump(
        {
            "model": best_model,
            "threshold": decision_threshold,
            "model_name": best_name,
            "selected_features": selected_features,
        },
        os.path.join(MODEL_DIR, "churn_model.pkl"),
    )
    
    return {
        "best_model": best_name,
        "best_params": best_search.best_params_,
        "cv_f1": float(best_search.best_score_),
        "model_search_results": model_search_results,
        "baseline_metrics": baseline_metrics,
        "test_metrics_at_0.5": default_metrics,
        "test_metrics_at_tuned_threshold": tuned_metrics,
        "decision_threshold": decision_threshold,
        "selected_features": selected_features,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "churn_rate": float(y.mean())
    }

if __name__ == "__main__":
    print("Running RFM segmentation & Churn prediction pipeline...")
    try:
        rfm, diag, prof, pca_var = rfm_and_segments()
        print("Segmentation diagnostics computed successfully.")
        print(diag)
        lapse_res = build_lapse_model()
        print("Lapse / Churn model trained successfully:", lapse_res)
    except Exception as e:
        print("Error executing model pipeline:", e)
