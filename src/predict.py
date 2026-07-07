from ultralytics import YOLO
from .constants import MODEL_NAME

model = YOLO(f"models/{MODEL_NAME}")

results = model("data/raw/raw1.png")

print(results)
results.show()