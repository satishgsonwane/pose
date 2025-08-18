from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model.track(source="long.mp4", show=True, save=True)  # predict on an image