from fastapi import FastAPI
from recurrent_health_events_prediction.api.routes.predict import router as predict_router

app = FastAPI(
    title="Hospital Readmission Prediction API",
    version="0.1.0",
)

@app.get("/")
def healthcheck():
    return {"status": "ok"}

app.include_router(predict_router)
