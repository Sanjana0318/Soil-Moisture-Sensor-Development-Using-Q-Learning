from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import asyncio
import random
from datetime import datetime
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from collections import deque
from sklearn.metrics import r2_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

class IoTFieldDevice:
    def __init__(self, field_id, location, crop_type):
        self.field_id = field_id
        self.location = location
        self.crop_type = crop_type
        
        # Set base parameters based on crop type
        if crop_type == "flower":
            # Flowers (e.g., roses) need moderate moisture and temperature
            self.base_moisture = 60 + random.uniform(-10, 10)
            self.base_temperature = 22 + random.uniform(-5, 5)
            self.base_light = 1000 + random.uniform(-200, 200)
            self.optimal_moisture = (50, 70)
            self.optimal_temp = (18, 26)
            self.optimal_light = (800, 1200)
            
        elif crop_type == "rice":
            # Rice needs high moisture and warm temperatures
            self.base_moisture = 80 + random.uniform(-5, 5)
            self.base_temperature = 28 + random.uniform(-3, 3)
            self.base_light = 900 + random.uniform(-100, 100)
            self.optimal_moisture = (75, 90)
            self.optimal_temp = (25, 32)
            self.optimal_light = (700, 1100)
            
        else:  # underground crop (potatoes)
            # Potatoes need moderate moisture and cooler temperatures
            self.base_moisture = 50 + random.uniform(-10, 10)
            self.base_temperature = 18 + random.uniform(-5, 5)
            self.base_light = 700 + random.uniform(-200, 200)
            self.optimal_moisture = (45, 65)
            self.optimal_temp = (15, 22)
            self.optimal_light = (600, 900)
            
        self.soil_type = random.choice(["clay", "loam", "sandy"])
        self.moisture_drift = random.uniform(-0.2, 0.2)
        self.temp_drift = random.uniform(-0.1, 0.1)
        self.health_status = "healthy"
        self.health_score = 100

    async def publish_sensor_data(self):
        timestamp = datetime.now()
        hour = timestamp.hour

        # Generate moisture data with crop-specific patterns
        moisture = self.base_moisture - 5 * np.sin(hour / 24 * 2 * np.pi) + random.uniform(-3, 3)
        moisture += self.moisture_drift
        moisture = max(10, min(90, moisture))

        # Generate temperature data with crop-specific patterns
        temp_variation = 8 if self.crop_type == "flower" else (5 if self.crop_type == "rice" else 6)
        temperature = self.base_temperature + temp_variation * np.sin((hour - 3) / 24 * 2 * np.pi) + random.uniform(-1, 1)
        temperature += self.temp_drift

        # Light patterns
        if 6 <= hour <= 18:
            light = self.base_light + 400 * np.sin((hour - 6) / 12 * np.pi) + random.uniform(-50, 50)
        else:
            light = self.base_light * 0.1 + random.uniform(-20, 20)

        humidity = 60 + 15 * np.sin(hour / 24 * 2 * np.pi) + random.uniform(-5, 5)
        soil_ph = 6.5 + random.uniform(-0.3, 0.3)
        
        # Calculate health score
        self.calculate_health(moisture, temperature, light)

        data = {
            "timestamp": timestamp.isoformat(),
            "field_id": self.field_id,
            "location": self.location,
            "crop_type": self.crop_type,
            "readings": {
                "moisture": round(moisture, 1),
                "temperature": round(temperature, 1),
                "light": round(light, 1),
                "humidity": round(humidity, 1),
                "soil_ph": round(soil_ph, 2)
            },
            "metadata": {
                "soil_type": self.soil_type,
                "health_status": self.health_status,
                "health_score": self.health_score
            }
        }
        return data
    
    def calculate_health(self, moisture, temperature, light):
        # Calculate deviations from optimal ranges
        moisture_dev = 0
        if moisture < self.optimal_moisture[0]:
            moisture_dev = (self.optimal_moisture[0] - moisture) / self.optimal_moisture[0]
        elif moisture > self.optimal_moisture[1]:
            moisture_dev = (moisture - self.optimal_moisture[1]) / (100 - self.optimal_moisture[1])
            
        temp_dev = 0
        if temperature < self.optimal_temp[0]:
            temp_dev = (self.optimal_temp[0] - temperature) / self.optimal_temp[0]
        elif temperature > self.optimal_temp[1]:
            temp_dev = (temperature - self.optimal_temp[1]) / (50 - self.optimal_temp[1])
            
        light_dev = 0
        if light < self.optimal_light[0]:
            light_dev = (self.optimal_light[0] - light) / self.optimal_light[0]
        elif light > self.optimal_light[1]:
            light_dev = (light - self.optimal_light[1]) / (2000 - self.optimal_light[1])
            
        # Calculate health score (0-100)
        self.health_score = max(0, 100 - (moisture_dev*30 + temp_dev*40 + light_dev*30))
        
        # Determine health status
        if self.health_score > 80:
            self.health_status = "healthy"
        elif self.health_score > 50:
            self.health_status = "moderate"
        else:
            self.health_status = "poor"

class MLModelManager:
    def __init__(self):
        self.models = {
            "decision_tree": DecisionTreeRegressor(max_depth=4, random_state=42),
            "svm": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.data_buffer = {
            "moisture": deque(maxlen=100),
            "temperature": deque(maxlen=100),
            "timestamps": deque(maxlen=100)
        }
        self.predictions = {
            "decision_tree": None,
            "svm": None,
            "gradient_boosting": None
        }
        self.last_training_time = None
        self.current_temp = None

    def add_data_point(self, timestamp, readings):
        try:
            ts_value = timestamp.timestamp()
            self.current_temp = readings.get('temperature')
            
            for key in ["moisture", "temperature"]:
                if key in readings:
                    self.data_buffer[key].append(readings[key])
            self.data_buffer["timestamps"].append(ts_value)

            if (len(self.data_buffer["moisture"]) >= 20 and 
                (self.last_training_time is None or 
                 (datetime.now() - self.last_training_time).total_seconds() > 30)):
                self.update_models()
        except Exception as e:
            print(f"Error adding data point: {e}")

    def update_models(self):
        try:
            min_length = min(
                len(self.data_buffer["moisture"]),
                len(self.data_buffer["temperature"]),
                len(self.data_buffer["timestamps"])
            )
            
            if min_length < 10:
                return
                
            moisture = np.array(list(self.data_buffer["moisture"])[-min_length:])
            temperature = np.array(list(self.data_buffer["temperature"])[-min_length:])
            timestamps = np.array(list(self.data_buffer["timestamps"])[-min_length:])
            
            hours = np.array([datetime.fromtimestamp(ts).hour for ts in timestamps])
            hours_sin = np.sin(2 * np.pi * hours / 24)
            hours_cos = np.cos(2 * np.pi * hours / 24)
            
            X = np.column_stack((temperature, hours_sin, hours_cos))
            y = moisture
            
            X_scaled = self.scaler.fit_transform(X)
            
            temp_min, temp_max = np.min(temperature), np.max(temperature)
            X_test_temp = np.linspace(temp_min - 2, temp_max + 2, 100)
            
            avg_hour_sin = np.mean(hours_sin)
            avg_hour_cos = np.mean(hours_cos)
            X_test = np.column_stack((
                X_test_temp,
                np.full_like(X_test_temp, avg_hour_sin),
                np.full_like(X_test_temp, avg_hour_cos)
            ))
            X_test_scaled = self.scaler.transform(X_test)
            
            for model_name, model in self.models.items():
                try:
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Get prediction for current temperature
                    current_X = np.array([[self.current_temp, avg_hour_sin, avg_hour_cos]])
                    current_X_scaled = self.scaler.transform(current_X)
                    current_pred = model.predict(current_X_scaled)[0]
                    
                    train_score = model.score(X_scaled, y)
                    
                    self.predictions[model_name] = {
                        "X": X_test_temp.tolist(),
                        "y": y_pred.tolist(),
                        "actual_X": temperature.tolist(),
                        "actual_y": moisture.tolist(),
                        "score": train_score,
                        "current_pred": round(current_pred, 1)
                    }
                    
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    self.predictions[model_name] = None
            
            self.last_training_time = datetime.now()
            self.publish_predictions()
            
        except Exception as e:
            print(f"Error updating ML models: {e}")

    def publish_predictions(self):
        try:
            predictions_to_send = {
                k: v for k, v in self.predictions.items() if v is not None
            }
            if predictions_to_send:
                socketio.emit('ml_predictions', predictions_to_send)
        except Exception as e:
            print(f"Error publishing predictions: {e}")

# Create fields with different crop types
fields = {
    "Flower Field": IoTFieldDevice("field1", (35.123, -97.456), "flower"),
    "Rice Field": IoTFieldDevice("field2", (35.135, -97.442), "rice"),
    "Potato Field": IoTFieldDevice("field3", (35.118, -97.461), "underground")
}

ml_manager = MLModelManager()
history_data = {field: {
    "moisture": [], "temperature": [], "light": [],
    "humidity": [], "soil_ph": [], "timestamps": [], "health": []
} for field in fields}

@app.route('/')
def index():
    return render_template('index.html', fields=list(fields.keys()))

@app.route('/api/field_data/<field_name>')
def get_field_data(field_name):
    if field_name in history_data:
        return jsonify({
            "status": "success",
            "data": history_data[field_name]
        })
    return jsonify({"status": "error", "message": "Field not found"}), 404

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_simulation')
def handle_start_simulation():
    socketio.start_background_task(target=collect_data)

@socketio.on('stop_simulation')
def handle_stop_simulation():
    global is_running
    is_running = False

@socketio.on('send_irrigation_command')
def handle_irrigation_command(data):
    field_name = data.get('field_name')
    if field_name in fields:
        command = {
            "command": "irrigate",
            "field_id": fields[field_name].field_id,
            "duration": 30,
            "timestamp": datetime.now().isoformat(),
            "sender": "web"
        }
        socketio.emit('irrigation_command', command)
        socketio.emit('log_message', {'message': f"Irrigation command sent to {field_name}", 'type': 'info'})

def collect_data():
    global is_running
    is_running = True
    while is_running:
        for field_name, device in fields.items():
            data = asyncio.run(device.publish_sensor_data())
            process_reading(field_name, data)
        socketio.sleep(5)

def process_reading(field_name, data):
    timestamp = datetime.fromisoformat(data.get("timestamp"))
    readings = data.get("readings", {})
    metadata = data.get("metadata", {})
    
    for key in ["moisture", "temperature", "light", "humidity", "soil_ph"]:
        if key in readings:
            history_data[field_name][key].append(readings[key])
    
    history_data[field_name]["timestamps"].append(timestamp)
    history_data[field_name]["health"].append(metadata.get("health_score", 100))
    ml_manager.add_data_point(timestamp, readings)
    
    # Check for alerts
    check_alerts(field_name, readings, metadata)
    
    socketio.emit('sensor_data', {
        'field_name': field_name,
        'data': {
            'timestamp': timestamp.isoformat(),
            'readings': readings,
            'metadata': metadata
        }
    })

def check_alerts(field_name, readings, metadata):
    device = fields[field_name]
    
    # Moisture alert
    if readings['moisture'] < device.optimal_moisture[0]:
        socketio.emit('alert', {
            'field': field_name,
            'type': 'moisture',
            'level': 'low',
            'value': readings['moisture'],
            'optimal': device.optimal_moisture,
            'message': f"Low moisture detected in {field_name} ({readings['moisture']}%)"
        })
    elif readings['moisture'] > device.optimal_moisture[1]:
        socketio.emit('alert', {
            'field': field_name,
            'type': 'moisture',
            'level': 'high',
            'value': readings['moisture'],
            'optimal': device.optimal_moisture,
            'message': f"High moisture detected in {field_name} ({readings['moisture']}%)"
        })
    
    # Temperature alert
    if readings['temperature'] < device.optimal_temp[0]:
        socketio.emit('alert', {
            'field': field_name,
            'type': 'temperature',
            'level': 'low',
            'value': readings['temperature'],
            'optimal': device.optimal_temp,
            'message': f"Low temperature detected in {field_name} ({readings['temperature']}°C)"
        })
    elif readings['temperature'] > device.optimal_temp[1]:
        socketio.emit('alert', {
            'field': field_name,
            'type': 'temperature',
            'level': 'high',
            'value': readings['temperature'],
            'optimal': device.optimal_temp,
            'message': f"High temperature detected in {field_name} ({readings['temperature']}°C)"
        })
    
    # Health status alert
    if metadata.get('health_status') == 'poor':
        socketio.emit('alert', {
            'field': field_name,
            'type': 'health',
            'level': 'critical',
            'value': metadata.get('health_score'),
            'optimal': (80, 100),
            'message': f"Poor health detected in {field_name} (score: {metadata.get('health_score')})"
        })

if __name__ == '__main__':
    socketio.run(app, port=8000, debug=True)