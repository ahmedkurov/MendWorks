from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

# -----------------------
# LSTM MODEL (Same as training)
# -----------------------
class MultiTaskLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=2, n_classes=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.cls_head = nn.Linear(hidden_dim // 2, n_classes)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        x = self.relu(self.fc(last))
        return self.cls_head(x), self.reg_head(x).squeeze(1)

# -----------------------
# PREDICTION CLASS
# -----------------------
class EEGFailurePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.ttf_scaler = None
        self.le = None
        self.feature_names = ["Temperature(°C)","Voltage(V)","Current(A)","Vibration(g)","Pressure(kPa)","Humidity(%)","Usage Hours(h)"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 30
        
    def load_model_and_preprocessors(self):
        """Load the trained model and all preprocessors"""
        try:
            # Get the directory where this script is located
            current_dir = Path(__file__).parent
            eegmodel_dir = current_dir.parent / "eegmodel"
            
            # Load preprocessors
            self.scaler = joblib.load(eegmodel_dir / "feature_scaler.pkl")
            self.ttf_scaler = joblib.load(eegmodel_dir / "ttf_scaler.pkl")
            self.le = joblib.load(eegmodel_dir / "label_encoder.pkl")
            
            # Initialize and load model
            self.model = MultiTaskLSTM(n_features=len(self.feature_names), n_classes=len(self.le.classes_))
            self.model.load_state_dict(torch.load(eegmodel_dir / "lstm_eeg_failure_model.pth", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model and preprocessors loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_from_dict(self, input_data):
        """Make prediction from input dictionary"""
        try:
            # Scale input
            X = self.scaler.transform(pd.DataFrame([input_data], columns=self.feature_names))
            seq = np.tile(X[:, None, :], (1, self.seq_len, 1))
            seq_t = torch.tensor(seq, dtype=torch.float32).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                cls_logits, reg_out = self.model(seq_t)
                probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
                pred_class = self.le.inverse_transform([np.argmax(probs)])[0]
                pred_ttf = self.ttf_scaler.inverse_transform([[reg_out.cpu().numpy()[0]]])[0][0]
            
            # Determine failure status and root cause
            failure = "Yes" if pred_class in ["Critical", "Warning"] else "No"
            cause, solution = self.analyze_failure_cause(input_data)
            
            # Format probability text
            prob_text = "\n".join([f"{cls}: {probs[i]*100:.1f}%" for i, cls in enumerate(self.le.classes_)])
            
            return {
                'condition': pred_class,
                'failure': failure,
                'cause': cause,
                'solution': solution,
                'time_to_failure': float(pred_ttf),
                'probabilities': prob_text,
                'confidence': float(np.max(probs) * 100)
            }
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def analyze_failure_cause(self, input_data):
        """Analyze the root cause of potential failure"""
        cause = "Sensor Fault"
        solution = "Recalibrate sensors"
        
        if input_data["Temperature(°C)"] > 65:
            cause, solution = "Overheating", "Check cooling system"
        elif input_data["Voltage(V)"] < 190 or input_data["Voltage(V)"] > 240:
            cause, solution = "Voltage Surge", "Stabilize power supply"
        elif input_data["Vibration(g)"] > 3:
            cause, solution = "Mechanical Wear", "Inspect moving parts"
        elif input_data["Usage Hours(h)"] > 8000:
            cause, solution = "Wear & Tear", "Replace worn components"
        
        return cause, solution

# Initialize predictor
predictor = EEGFailurePredictor()
predictor.load_model_and_preprocessors()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'EEG Prediction API is running'})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input data
        required_fields = ["temperature", "voltage", "current", "vibration", "pressure", "humidity", "usageHours"]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Convert to the format expected by the model
        input_data = {
            "Temperature(°C)": float(data["temperature"]),
            "Voltage(V)": float(data["voltage"]),
            "Current(A)": float(data["current"]),
            "Vibration(g)": float(data["vibration"]),
            "Pressure(kPa)": float(data["pressure"]),
            "Humidity(%)": float(data["humidity"]),
            "Usage Hours(h)": float(data["usageHours"])
        }
        
        # Make prediction
        result = predictor.predict_from_dict(input_data)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
