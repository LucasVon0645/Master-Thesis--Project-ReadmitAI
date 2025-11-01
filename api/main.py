from fastapi import FastAPI
from api.routes.predict import router as predict_router
from api.routes.explain_single_patient import router as explain_single_patient

app = FastAPI(
    title="Hospital Readmission Prediction API",
    version="0.1.0",
)

@app.get("/health")
def healthcheck():
    return {"status": "ok"}

app.include_router(predict_router)
app.include_router(explain_single_patient)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app",
                host="0.0.0.0", port=8000, reload=True)