# Carla Autonomy

Autonomous driving project for the [carla simulator.](https://carla.org)

Currently implemented:
-MobileNetV3 based model
-Semantic segmentation to identify current lane boundaries
-Custom hybrid Tversky/Cross-entropy loss function
-Precision, recall, and dice metrics per boundary
-Prediction and confidence map visualization

Next up:
-Lane geometry estimation
-Calculate steering angle

## Setup
```
pip install -r requirements.txt
```
Paths for training data and outputs and other settings can be edited in config.py.