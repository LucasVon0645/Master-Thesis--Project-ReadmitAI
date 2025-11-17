import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    # Load example payload
    with open("api/docs/example_payload_predict.json", "r") as f:
        example_payload = json.load(f)

    response = client.post("/predict", json=example_payload)
    assert response.status_code == 200

    body = response.json()
    # Root has: prediction, metadata, metrics (maybe null)
    assert "prediction" in body
    assert "metadata" in body
    assert "metrics" in body  # can still be null

    prediction = body["prediction"]
    # Check required fields of PredictionBody
    assert "pred_probs" in prediction
    assert "pred_labels" in prediction
    assert "hadm_ids" in prediction
    assert "subject_ids" in prediction
    
    assert set(prediction["subject_ids"]) == set([222, 773])  # Example check
    assert set(prediction["hadm_ids"]) == set([900002, 160425])  # Example check
    
    # Sanity checks on shapes
    probs = prediction["pred_probs"]
    labels = prediction["pred_labels"]
    hadm_ids = prediction["hadm_ids"]
    attention_weights = prediction.get("attention_weights", None)

    assert len(probs) == len(labels) == len(hadm_ids)
    if attention_weights is not None:
        assert len(attention_weights) == len(hadm_ids)
    
    # Check fields in metadata
    metadata = body["metadata"]
    assert 'model_name' in metadata
    assert 'number_of_predictions' in metadata
    assert 'timestamp' in metadata
    assert 'prob_threshold' in metadata
    
    number_of_predictions = metadata['number_of_predictions']
    assert number_of_predictions == len(labels)
    prob_threshold = metadata['prob_threshold']
    assert 0.0 <= prob_threshold <= 1.0
    
    # Check fields in metrics (can be null)
    metrics = body["metrics"]
    if metrics is not None:
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics
        assert 'confusion_matrix' in metrics