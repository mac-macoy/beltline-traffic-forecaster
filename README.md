## beltline-traffic-forecaster
### Summary
Goal: Predict the traffic on the Beltline at any given time

Strategy:
1. Gather traffic data from Beltline video using human detection
2. Gather any other possibly useful data (weather, etc.)
3. Train a model to predict traffic levels at a given time

### Setup
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip3 install -r requirements.txt`
