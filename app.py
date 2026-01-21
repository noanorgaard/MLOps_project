from fastapi import FastAPI
import torch
from src.ai_vs_human.model import get_model  # or MyModel

app = FastAPI()

MODEL_PATH = "models/model.pth"

model = get_model()  # same function used in training
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(x: list[float]):
    with torch.no_grad():
        x_tensor = torch.tensor(x).unsqueeze(0)
        y = model(x_tensor)
    return {"prediction": y.tolist()}
