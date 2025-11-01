from functools import lru_cache
from api.services.prediction import ModelPrediction


@lru_cache(maxsize=1)
def get_service() -> ModelPrediction:
    """
    Reuse a single ModelPrediction instance per process.
    (Loads model/scaler lazily on first use.)
    """
    return ModelPrediction()