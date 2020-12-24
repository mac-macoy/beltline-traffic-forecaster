## beltline-traffic-forecaster
### Summary
Goal: Predict the traffic on the Beltline at any given time

Strategy:
1. Gather traffic data from Beltline video using human detection
2. Gather any other possibly useful data (weather, etc.)
3. Train a model to predict traffic levels at a given time

Data Collection: See [detect_traffic.py](https://github.com/mac-macoy/beltline-traffic-forecaster/blob/master/detect_traffic.py)

Prediction: See [predict_traffic.ipynb](https://github.com/mac-macoy/beltline-traffic-forecaster/blob/master/predict_traffic.ipynb)