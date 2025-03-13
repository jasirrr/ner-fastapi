from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import joblib
import os
import logging

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Load trained CRF model
# ✅ Load trained CRF model from container path
MODEL_PATH = "/app/ner_crf_model.pkl"
crf_model = joblib.load(MODEL_PATH)


# ✅ Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ner-api")

# ✅ Authentication setup
security = HTTPBasic()

USERNAME = os.getenv("API_USERNAME", "admin")
PASSWORD = os.getenv("API_PASSWORD", "secret")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username == USERNAME and credentials.password == PASSWORD:
        return credentials.username
    raise HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

# ✅ Request model
class TextInput(BaseModel):
    text: str

# ✅ Health check endpoint
@app.get("/health/")
def health():
    return {"status": "ok"}

# ✅ Prediction endpoint
@app.post("/predict/")
def predict(input_data: TextInput, user: str = Depends(get_current_user)):
    try:
        logger.info(f"Received request from user {user}")
        
        # Tokenize text
        tokens = input_data.text.split()

        # Generate features for CRF
        features = [{"word": token} for token in tokens]
        
        # Predict entities
        predicted_entities = crf_model.predict([features])[0]
        
        response = [{"text": token, "label": label} for token, label in zip(tokens, predicted_entities)]

        return {"entities": response}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# ✅ Run FastAPI locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
