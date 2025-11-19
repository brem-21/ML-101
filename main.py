from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import arff
import joblib
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os

app = FastAPI(title="Click Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: list[float]

class TrainingResponse(BaseModel):
    message: str
    accuracy: float
    best_params: dict

# Global variables to store model and feature selector
model = None
feature_selector = None
feature_names = None

def load_data():
    """Load and preprocess the click data"""
    url = 'https://www.openml.org/data/download/184157/phpfGCaQC'
    response = requests.get(url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to download data")
    
    arff_data = arff.loads(response.text)
    clicks = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
    clicks['click'] = clicks['click'].astype(int)
    
    return clicks

@app.post("/train", response_model=TrainingResponse)
async def train_model():
    """Train the XGBoost model with hyperparameter tuning"""
    global model, feature_selector, feature_names
    
    try:
        # Load data
        clicks = load_data()
        X = clicks.drop('click', axis=1)
        y = clicks['click']
        
        # Feature selection
        feature_selector = SelectKBest(mutual_info_classif, k=5)
        X_selected = feature_selector.fit_transform(X, y)
        feature_names = X.columns[feature_selector.get_support(indices=True)].tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }
        
        xgb_model = xgb.XGBClassifier()
        grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best model and evaluate
        model = grid_search.best_estimator_
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model and feature selector
        joblib.dump(model, 'best_model.pkl')
        joblib.dump(feature_selector, 'feature_selector.pkl')
        
        return TrainingResponse(
            message="Model trained successfully",
            accuracy=round(accuracy, 4),
            best_params=grid_search.best_params_
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using the trained model"""
    global model, feature_selector
    
    if model is None:
        # Try to load existing model
        if os.path.exists('best_model.pkl') and os.path.exists('feature_selector.pkl'):
            model = joblib.load('best_model.pkl')
            feature_selector = joblib.load('feature_selector.pkl')
        else:
            raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    try:
        # Convert input to numpy array and reshape
        features = np.array(request.features).reshape(1, -1)
        
        # Apply feature selection if needed
        if len(request.features) > 5:
            features = feature_selector.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "message": "Click" if prediction == 1 else "No Click"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the current model"""
    global model, feature_names
    
    if model is None:
        return {"message": "No model trained"}
    
    return {
        "model_type": "XGBoost Classifier",
        "selected_features": feature_names,
        "feature_count": len(feature_names) if feature_names else 0
    }

@app.get("/")
async def root():
    return {"message": "Click Prediction API", "endpoints": ["/train", "/predict", "/model-info"]}