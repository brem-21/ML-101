import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_train_model():
    """Test the model training endpoint"""
    print("Testing model training...")
    response = requests.post(f"{BASE_URL}/train")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Training successful!")
        print(f"   Accuracy: {result['accuracy']}")
        print(f"   Best params: {result['best_params']}")
    else:
        print(f"❌ Training failed: {response.text}")

def test_prediction():
    """Test the prediction endpoint"""
    print("\nTesting prediction...")
    
    # Sample features (5 values for the selected features)
    test_data = {
        "features": [0.5, 1.2, 0.8, 2.1, 0.3]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data)
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"   Prediction: {result['prediction']} ({result['message']})")
        print(f"   Probability: {result['probability']}")
    else:
        print(f"❌ Prediction failed: {response.text}")

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info...")
    response = requests.get(f"{BASE_URL}/model-info")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Model info retrieved!")
        print(f"   Model type: {result.get('model_type', 'N/A')}")
        print(f"   Selected features: {result.get('selected_features', 'N/A')}")
    else:
        print(f"❌ Model info failed: {response.text}")

if __name__ == "__main__":
    print("🚀 Testing Click Prediction API")
    print("=" * 40)
    
    # Test all endpoints
    test_train_model()
    test_prediction()
    test_model_info()
    
    print("\n" + "=" * 40)
    print("✨ Testing complete!")