import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_explain_patient_endpoint():
    # Load example payload
    with open("api/docs/example_payload_single_patient.json", "r") as f:
        example_payload = json.load(f)

    response = client.post("/explain_single_patient", json=example_payload)
    assert response.status_code == 200

    body = response.json()
    assert "explanation" in body
    assert "input_features" in body
    assert "metadata" in body

    explanation = body["explanation"]
    assert "feature_attribution_split" in explanation
    assert "past_features_attributions" in explanation
    assert "current_features_attributions" in explanation
    
    # Check feature attribution split structure
    feature_attribution_split = explanation["feature_attribution_split"]
    assert "past_attribution" in feature_attribution_split
    assert isinstance(feature_attribution_split["past_attribution"], float)
    assert "current_attribution" in feature_attribution_split
    assert isinstance(feature_attribution_split["current_attribution"], float)
    
    # Check current features attributions structure
    current_features_attributions = explanation["current_features_attributions"]
    assert isinstance(current_features_attributions, list)
    assert all(isinstance(attr, dict) for attr in current_features_attributions)
    
    # Check past features attributions structure
    past_features_attributions = explanation["past_features_attributions"]
    assert isinstance(past_features_attributions, list)
    assert all(isinstance(attr, dict) for attr in past_features_attributions)
    
    # Check input features structure
    input_features = body["input_features"]
    assert "past" in input_features
    assert "current" in input_features
    current_features = input_features["current"]
    assert isinstance(current_features, dict)
    past_features = input_features["past"]
    assert isinstance(past_features, list)
    assert all(isinstance(feat, dict) for feat in past_features)
    
    # Check metadata structure
    metadata = body["metadata"]
    assert 'model_name' in metadata
    assert 'number_of_predictions' in metadata
    assert 'timestamp' in metadata
    assert 'prob_threshold' in metadata
    
    number_of_predictions = metadata['number_of_predictions']
    assert number_of_predictions == 1  # single patient