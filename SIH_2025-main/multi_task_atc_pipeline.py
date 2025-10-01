"""
ATC Multi-Task Pipeline + Cow Classifier
----------------------------------------
Predict cattle longevity, milk productivity, reproductive efficiency, and elite dam status
from both tabular and optional image inputs.

Key Features:
1. Handles missing values via median imputation.
2. Uses PIL for image processing to avoid OpenCV dependency.
3. Optional CNN feature extraction with ResNet50 (local ImageNet weights) if TensorFlow is available.
4. Cow vs Non-Cow image validation using custom CNN classifier.
5. Trains four separate models (three regressors, one classifier) using Random Forests.
6. Command-line interface supports `train` and `predict` modes.
7. Computes an ATP Score (0–100) combining longevity, milk yield, reproductive efficiency, and elite probability.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Optional   # ✅ Python 3.8 compatibility

# ---------------------- TensorFlow imports ----------------------
try:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ---------------------- PIL imports ----------------------
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ---------------------- CONFIGURATION ----------------------
DATA_PATH                = "data/cow_data.csv"
LON_MODEL_PATH           = "models/longevity_model.pkl"
MILK_MODEL_PATH          = "models/milk_productivity_model.pkl"
REPRO_MODEL_PATH         = "models/reproductive_efficiency_model.pkl"
ELITE_MODEL_PATH         = "models/elite_dam_model.pkl"
COW_CLASSIFIER_PATH      = "models/cow_classifier.h5"   # CNN cow detector
CALIBRATION_CM_PER_PIXEL = 0.1  # 1 pixel = 0.1 cm

RESNET50_WEIGHTS_PATH    = r"C:\Users\SIVARAMAN\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------- CNN LOADER -------------------------
CNN_MODEL = None
COW_CLASSIFIER = None

def get_cnn_model():
    """Load ResNet50 for feature extraction lazily only if image is used."""
    global CNN_MODEL
    if CNN_MODEL is None and TF_AVAILABLE:
        if os.path.exists(RESNET50_WEIGHTS_PATH):
            print(f"Loading ResNet50 from local weights: {RESNET50_WEIGHTS_PATH}")
            CNN_MODEL = ResNet50(weights=RESNET50_WEIGHTS_PATH, include_top=False, pooling="avg")
        else:
            print("⚠️ Local ResNet50 weights not found, downloading ImageNet weights.")
            CNN_MODEL = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return CNN_MODEL

def get_cow_classifier():
    """Load CNN cow classifier if available."""
    global COW_CLASSIFIER
    if COW_CLASSIFIER is None and TF_AVAILABLE and os.path.exists(COW_CLASSIFIER_PATH):
        COW_CLASSIFIER = load_model(COW_CLASSIFIER_PATH)
    return COW_CLASSIFIER

# ---------------------- DATASET UTILITIES ------------------

def create_sample_dataset(path: str, n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "age":                  np.random.normal(5, 2, n_samples).clip(1, 12),
        "body_weight":          np.random.normal(500, 80, n_samples).clip(300, 800),
        "height_at_withers":    np.random.normal(140, 10, n_samples).clip(120, 160),
        "body_length":          np.random.normal(160, 15, n_samples).clip(130, 200),
        "chest_width":          np.random.normal(45, 8, n_samples).clip(30, 65),
        "parity":               np.random.poisson(2, n_samples).clip(0, 8),
        "historical_milk_yield":np.random.normal(6000, 1500, n_samples).clip(2000, 12000)
    }
    data["longevity"]              = (8 + 0.3*data['age'] - 0.002*data['body_weight'] + np.random.normal(0,1,n_samples)).clip(2,15)
    data["milk_productivity"]      = (4000 + 5*data['body_weight'] + 20*data['height_at_withers'] - 200*data['age'] + np.random.normal(0,500,n_samples)).clip(1000,15000)
    data["reproductive_efficiency"] = (1.2 + 0.1*data['age'] + np.random.normal(0,0.3,n_samples)).clip(0.8,4)
    data["elite_dam"]              = (data['milk_productivity'] > np.percentile(data['milk_productivity'], 75)).astype(int)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df

def load_and_clean_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Dataset not found at {path}. Creating synthetic dataset.")
        return create_sample_dataset(path)
    df = pd.read_csv(path)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    return df

# ---------------------- IMAGE UTILITIES --------------------

def load_image(path: str):
    if not (PIL_AVAILABLE and os.path.exists(path)):
        return None
    return Image.open(path).convert("RGB")

def preprocess_image(img):
    return img.resize((800, 600)) if img else None

def detect_landmarks(img):
    if img is None:
        return {}
    w, h = img.size
    return {
        "withers":      (int(w*0.5), int(h*0.3)),
        "rump":         (int(w*0.5), int(h*0.6)),
        "chest_left":   (int(w*0.4), int(h*0.45)),
        "chest_right":  (int(w*0.6), int(h*0.45))
    }

def px2cm(px):
    return px * CALIBRATION_CM_PER_PIXEL

def compute_measurements(lm):
    if not lm:
        return {"height_at_withers":140, "body_length":160, "chest_width":45}
    w, r, cl, cr = lm["withers"], lm["rump"], lm["chest_left"], lm["chest_right"]
    height_px = abs(r[1]-w[1])
    body_px   = ((w[0]-r[0])**2 + (w[1]-r[1])**2)**0.5
    chest_px  = ((cl[0]-cr[0])**2 + (cl[1]-cr[1])**2)**0.5
    return {
        "height_at_withers": px2cm(height_px),
        "body_length":       px2cm(body_px),
        "chest_width":       px2cm(chest_px)
    }

def extract_cnn_features(img_path: str):
    model = get_cnn_model()
    if not model:
        return np.zeros(2048)
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feats = model.predict(x, verbose=0)
    return feats.flatten()

def extract_image_features(img_path: str):
    feats = {"height_at_withers":140, "body_length":160, "chest_width":45, "cnn_features":np.zeros(2048)}
    img = load_image(img_path)
    if img is None:
        return feats
    img_proc = preprocess_image(img)
    lm      = detect_landmarks(img_proc)
    feats.update(compute_measurements(lm))
    feats["cnn_features"] = extract_cnn_features(img_path)
    return feats

def is_cow_image(img_path: str) -> bool:
    """Check if uploaded image is a cow using CNN classifier."""
    clf = get_cow_classifier()
    if not clf:
        return True  # fallback
    img = keras_image.load_img(img_path, target_size=(128, 128))
    x = keras_image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = clf.predict(x, verbose=0)[0][0]
    return pred > 0.5

# ---------------------- MODEL TRAINING ---------------------

def train_models(df: pd.DataFrame):
    base_features = [
        "age", "body_weight", "height_at_withers", "body_length", "chest_width", "parity", "historical_milk_yield"
    ]
    X = df[base_features]

    def _save(model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

    if "longevity" in df:
        y = df["longevity"]
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        print(f"Longevity R²: {m.score(X_val,y_val):.3f}")
        _save(m, LON_MODEL_PATH)

    if "milk_productivity" in df:
        y = df["milk_productivity"]
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        print(f"Milk productivity R²: {m.score(X_val,y_val):.3f}")
        _save(m, MILK_MODEL_PATH)

    if "reproductive_efficiency" in df:
        y = df["reproductive_efficiency"]
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        print(f"Reproductive efficiency R²: {m.score(X_val,y_val):.3f}")
        _save(m, REPRO_MODEL_PATH)

    if "elite_dam" in df:
        y = df["elite_dam"]
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        print(f"Elite dam accuracy: {m.score(X_val,y_val):.3f}")
        _save(m, ELITE_MODEL_PATH)

    print("✅ Training complete! Models saved in 'models/'.")

# ---------------------- PREDICTION -------------------------

def predict_attributes(age: float, body_weight: float, parity: int,
                       historical_milk_yield: float,
                       image_path: Optional[str] = None):
    if image_path and not is_cow_image(image_path):
        return {"error": "🚫 The uploaded image is not a cow. Please provide a valid cow image."}

    feats = {
        "age": age,
        "body_weight": body_weight,
        "parity": parity,
        "historical_milk_yield": historical_milk_yield,
        "height_at_withers": 140.0,
        "body_length": 160.0,
        "chest_width": 45.0
    }
    if image_path:
        img_feats = extract_image_features(image_path)
        feats.update({k: img_feats[k] for k in ("height_at_withers","body_length","chest_width")})

    order = ["age","body_weight","height_at_withers","body_length","chest_width","parity","historical_milk_yield"]
    X = pd.DataFrame([[feats[c] for c in order]], columns=order)

    outputs = {}
    if os.path.exists(LON_MODEL_PATH):
        outputs["longevity"] = joblib.load(LON_MODEL_PATH).predict(X)[0]
    if os.path.exists(MILK_MODEL_PATH):
        outputs["milk_productivity"] = joblib.load(MILK_MODEL_PATH).predict(X)[0]
    if os.path.exists(REPRO_MODEL_PATH):
        outputs["reproductive_efficiency"] = joblib.load(REPRO_MODEL_PATH).predict(X)[0]
    if os.path.exists(ELITE_MODEL_PATH):
        clf = joblib.load(ELITE_MODEL_PATH)
        outputs["elite_dam"] = clf.predict(X)[0]
        outputs["elite_dam_probability"] = clf.predict_proba(X)[0][1]

    if all(k in outputs for k in ["longevity", "milk_productivity", "reproductive_efficiency", "elite_dam_probability"]):
        longevity = outputs["longevity"]
        milk = outputs["milk_productivity"]
        repro = outputs["reproductive_efficiency"]
        elite_prob = outputs["elite_dam_probability"]

        atp_score = (
            (longevity / 15.0) * 0.25 +
            (milk / 8000.0) * 0.35 +
            (repro / 3.0) * 0.20 +
            (elite_prob) * 0.20
        ) * 100
        outputs["ATP_score"] = round(atp_score, 2)

    return outputs

# ---------------------- CLI ENTRY --------------------------
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Multi-task ATC pipeline")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--csv", type=str, required=True, help="Path to training dataset CSV")

    predict_parser = subparsers.add_parser("predict", help="Predict using trained models")
    predict_parser.add_argument("--image", type=str, help="Path to cow image")
    predict_parser.add_argument("--age", type=float, required=True)
    predict_parser.add_argument("--body_weight", type=float, required=True)
    predict_parser.add_argument("--parity", type=int, required=True)
    predict_parser.add_argument("--historical_milk_yield", type=float, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        print(f"Training with CSV: {args.csv}")
        df = load_and_clean_dataset(args.csv)
        train_models(df)

    elif args.mode == "predict":
        print("Predicting with inputs...")
        results = predict_attributes(
            age=args.age,
            body_weight=args.body_weight,
            parity=args.parity,
            historical_milk_yield=args.historical_milk_yield,
            image_path=args.image
        )
        print("Prediction Results:", results)
