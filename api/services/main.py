from functools import lru_cache

import yaml
from api.services.prediction import ModelPrediction
from importlib import resources as impresources
import api.configs as configs

with open(impresources.files(configs) / "config.yaml", encoding="utf-8") as f:
    api_config = yaml.safe_load(f)

@lru_cache(maxsize=1)
def get_service() -> ModelPrediction:
    """
    Reuse a single ModelPrediction instance per process.
    (Loads model/scaler lazily on first use.)
    """
    return ModelPrediction(api_config=api_config)