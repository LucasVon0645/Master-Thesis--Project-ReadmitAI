from fastapi import FastAPI
from api.routes.predict import router as predict_router

app = FastAPI(
    title="Hospital Readmission Prediction API",
    version="0.1.0",
)

@app.get("/")
def healthcheck():
    return {"status": "ok"}

app.include_router(predict_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recurrent_health_events_prediction.api.main:app",
                host="0.0.0.0", port=8000, reload=True)