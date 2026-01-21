import torch
from src.ai_vs_human.model import get_model   # <-- adjust this import
#from src.config import cfg     # <-- if you use configs

MODEL_PATH = "models/model.pth"
OUT_PATH = "models/model_full.pth"

model = get_model()           # <-- same args as training
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

torch.save(model, OUT_PATH)

print("Saved full model to", OUT_PATH)
