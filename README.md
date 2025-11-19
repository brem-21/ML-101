# Click Prediction API

## Problem Description

This project addresses the challenge of **predicting user click behavior** in digital advertising and web analytics. Click prediction is crucial for:

- **Ad Revenue Optimization**: Predicting which ads users are likely to click
- **Content Recommendation**: Improving user engagement by showing relevant content
- **Marketing Campaign Efficiency**: Targeting users with higher click probability
- **Resource Allocation**: Focusing computational resources on high-value predictions

The original problem involves analyzing user interaction data to build a machine learning model that can accurately predict whether a user will click on a specific item based on various features.

## Solution Overview

This FastAPI application transforms a complex machine learning pipeline into a production-ready web service that:

1. **Automates Data Processing**: Fetches and preprocesses click data from OpenML
2. **Performs Feature Selection**: Uses mutual information to identify the 5 most predictive features
3. **Optimizes Model Performance**: Implements hyperparameter tuning with GridSearchCV
4. **Provides Real-time Predictions**: Offers instant click probability predictions via REST API
5. **Ensures Model Persistence**: Saves trained models for consistent predictions

## Key Features

- **Automated Training Pipeline**: Complete ML workflow from data loading to model optimization
- **Feature Selection**: Intelligent feature reduction using mutual information criteria
- **Hyperparameter Tuning**: Grid search optimization for XGBoost classifier
- **RESTful API**: Clean, documented endpoints for training and prediction
- **Model Persistence**: Automatic saving/loading of trained models
- **Error Handling**: Comprehensive error management and validation

## API Endpoints

### 1. Train Model
```
POST /train
```
Trains the XGBoost model with hyperparameter optimization.

**Response:**
```json
{
  "message": "Model trained successfully",
  "accuracy": 0.8542,
  "best_params": {
    "learning_rate": 0.1,
    "max_depth": 4,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 1
  }
}
```

### 2. Make Prediction
```
POST /predict
```
Predicts click probability for given features.

**Request Body:**
```json
{
  "features": [0.5, 1.2, 0.8, 2.1, 0.3]
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.7834,
  "message": "Click"
}
```

### 3. Model Information
```
GET /model-info
```
Returns current model details and selected features.

## Installation & Setup

### Option 1: Local Installation

1. **Clone and Navigate:**
```bash
cd /path/to/project
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Application:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker (Recommended)

1. **Build Docker Image:**
```bash
docker build -t click-prediction-api .
```

2. **Run Docker Container:**
```bash
docker run -p 8000:8000 click-prediction-api
```

### Access the API:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## Usage Example

1. **Train the Model:**
```bash
curl -X POST "http://localhost:8000/train"
```

2. **Make a Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, 0.8, 2.1, 0.3]}'
```

## Technical Architecture

- **Framework**: FastAPI for high-performance async API
- **ML Library**: XGBoost for gradient boosting classification
- **Feature Selection**: Scikit-learn's mutual information classifier
- **Data Processing**: Pandas and NumPy for efficient data manipulation
- **Model Persistence**: Joblib for model serialization
- **Validation**: Pydantic for request/response validation

## Model Performance

The XGBoost classifier with hyperparameter tuning typically achieves:
- **Accuracy**: 85%+ on test data
- **Feature Reduction**: From all features to top 5 most predictive
- **Cross-validation**: 3-fold CV for robust parameter selection

## Production Considerations

- Models are automatically saved and can be reloaded on server restart
- Feature selection ensures consistent input dimensionality
- Comprehensive error handling for robust production deployment
- Async endpoints for handling concurrent requests efficiently

## Future Enhancements

- Model versioning and A/B testing capabilities
- Real-time model retraining with new data
- Advanced feature engineering pipelines
- Integration with monitoring and logging systems