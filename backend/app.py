from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import json
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file (in project root)
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

app = Flask(__name__)
CORS(app)

# -----------------------
# SUPABASE CONFIGURATION
# -----------------------
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://placeholder.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBsYWNlaG9sZGVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NDUxOTI4MDAsImV4cCI6MTk2MDc2ODgwMH0.placeholder')

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# LSTM MODELS (EEG, MRI, Ventilator)
# -----------------------
class EEGMultiTaskLSTM(nn.Module):
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

class MRIMultiTaskLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=128, n_layers=3, n_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=n_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim // 2)
        self.cls_head = nn.Linear(hidden_dim // 2, n_classes)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)
        last_hidden = attn_out[:, -1, :]
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm(x)
        cls_logits = self.cls_head(x)
        reg_out = self.reg_head(x).squeeze(1)
        confidence = torch.sigmoid(self.confidence_head(x)).squeeze(1)
        return cls_logits, reg_out, confidence

class VentilatorMultiTaskLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=128, n_layers=3, n_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=n_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim // 2)
        self.cls_head = nn.Linear(hidden_dim // 2, n_classes)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)
        last_hidden = attn_out[:, -1, :]
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm(x)
        cls_logits = self.cls_head(x)
        reg_out = self.reg_head(x).squeeze(1)
        confidence = torch.sigmoid(self.confidence_head(x)).squeeze(1)
        return cls_logits, reg_out, confidence

# -----------------------
# PREDICTION CLASSES
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
        """Load the trained EEG model and all preprocessors"""
        try:
            current_dir = Path(__file__).parent
            eegmodel_dir = current_dir.parent / "eegmodel"
            
            self.scaler = joblib.load(eegmodel_dir / "feature_scaler.pkl")
            self.ttf_scaler = joblib.load(eegmodel_dir / "ttf_scaler.pkl")
            self.le = joblib.load(eegmodel_dir / "label_encoder.pkl")
            
            self.model = EEGMultiTaskLSTM(n_features=len(self.feature_names), n_classes=len(self.le.classes_))
            self.model.load_state_dict(torch.load(eegmodel_dir / "lstm_eeg_failure_model.pth", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ EEG Model and preprocessors loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading EEG model: {e}")
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
            print(f"Error in EEG prediction: {e}")
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

class MRIFailurePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.ttf_scaler = None
        self.le = None
        self.feature_names = [
            "Helium_Level_pct", "Magnetic_Field_T", "RF_Power_kW", "Gradient_Temp_C",
            "Compressor_Pressure_bar", "Chiller_Flow_lpm", "Room_Humidity_pct", 
            "System_Runtime_hrs", "Vibration_mm_s", "Coil_Temperature_C", 
            "Power_Consumption_kW", "Magnet_Quench_Risk"
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 30
        
    def load_model_and_preprocessors(self):
        """Load the trained MRI model and all preprocessors"""
        try:
            current_dir = Path(__file__).parent
            allmodels_dir = current_dir.parent / "allmodels" / "mri model"
            
            self.scaler = joblib.load(allmodels_dir / "mri_feature_scaler.pkl")
            self.ttf_scaler = joblib.load(allmodels_dir / "mri_ttf_scaler.pkl")
            self.le = joblib.load(allmodels_dir / "mri_label_encoder.pkl")
            
            self.model = MRIMultiTaskLSTM(n_features=len(self.feature_names), n_classes=len(self.le.classes_))
            self.model.load_state_dict(torch.load(allmodels_dir / "lstm_mri_failure_model.pth", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ MRI Model and preprocessors loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading MRI model: {e}")
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
                cls_logits, reg_out, confidence = self.model(seq_t)
                probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
                pred_class = self.le.inverse_transform([np.argmax(probs)])[0]
                pred_ttf = self.ttf_scaler.inverse_transform([[reg_out.cpu().numpy()[0]]])[0][0]
                confidence_score = confidence.cpu().numpy()[0]
            
            # Determine failure status and root cause
            failure = "Yes" if pred_class in ["Critical", "Warning"] else "No"
            cause, solution = self.analyze_mri_failure_cause(input_data)
            
            # Format probability text
            prob_text = "\n".join([f"{cls}: {probs[i]*100:.1f}%" for i, cls in enumerate(self.le.classes_)])
            
            return {
                'condition': pred_class,
                'failure': failure,
                'cause': cause,
                'solution': solution,
                'time_to_failure': float(pred_ttf),
                'probabilities': prob_text,
                'confidence': float(confidence_score * 100)
            }
        except Exception as e:
            print(f"Error in MRI prediction: {e}")
            return None

    def analyze_mri_failure_cause(self, input_data):
        """Analyze the root cause of potential MRI failure"""
        cause = "System Degradation"
        solution = "Schedule comprehensive inspection"
        
        if input_data["Helium_Level_pct"] < 75 or input_data["Compressor_Pressure_bar"] < 10:
            cause = "Cryogenic System Failure"
            solution = "Check helium supply system and compressor"
        elif input_data["Magnetic_Field_T"] < 2.90 or input_data["Magnet_Quench_Risk"] > 0.3:
            cause = "Magnetic Field Instability"
            solution = "Inspect superconducting magnet and quench protection"
        elif input_data["RF_Power_kW"] > 45 or input_data["Coil_Temperature_C"] > 40:
            cause = "RF System Overheating"
            solution = "Service RF amplifiers and cooling system"
        elif input_data["Gradient_Temp_C"] > 35 or input_data["Chiller_Flow_lpm"] < 35:
            cause = "Gradient System Failure"
            solution = "Inspect gradient coils and cooling circuits"
        elif input_data["Power_Consumption_kW"] > 75:
            cause = "Power System Anomaly"
            solution = "Check power supply and electrical connections"
        elif input_data["Vibration_mm_s"] > 4.0 or input_data["System_Runtime_hrs"] > 10000:
            cause = "Mechanical Wear"
            solution = "Schedule mechanical maintenance and component replacement"
        elif input_data["Room_Humidity_pct"] > 70:
            cause = "Environmental Control Issues"
            solution = "Adjust HVAC system and environmental controls"
        
        return cause, solution

class VentilatorFailurePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.ttf_scaler = None
        self.le = None
        self.feature_names = [
            "Airway_Pressure_cmH2O", "Tidal_Volume_ml", "Respiratory_Rate_bpm",
            "Oxygen_Concentration_pct", "Inspiratory_Flow_lpm", "Exhaled_CO2_mmHg",
            "Humidifier_Temperature_C", "Compressor_Pressure_bar", "Valve_Response_ms",
            "Battery_Level_pct", "Power_Consumption_kW", "Alarm_Frequency_count",
            "System_Runtime_hrs", "Vibration_mm_s", "Internal_Temperature_C", "Filter_Status_pct"
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 30
        
    def load_model_and_preprocessors(self):
        """Load the trained Ventilator model and all preprocessors"""
        try:
            current_dir = Path(__file__).parent
            allmodels_dir = current_dir.parent / "allmodels" / "mri model"
            
            self.scaler = joblib.load(allmodels_dir / "ventilator_feature_scaler.pkl")
            self.ttf_scaler = joblib.load(allmodels_dir / "ventilator_ttf_scaler.pkl")
            self.le = joblib.load(allmodels_dir / "ventilator_label_encoder.pkl")
            
            self.model = VentilatorMultiTaskLSTM(n_features=len(self.feature_names), n_classes=len(self.le.classes_))
            self.model.load_state_dict(torch.load(allmodels_dir / "lstm_ventilator_failure_model.pth", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Ventilator Model and preprocessors loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Ventilator model: {e}")
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
                cls_logits, reg_out, confidence = self.model(seq_t)
                probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
                pred_class = self.le.inverse_transform([np.argmax(probs)])[0]
                pred_ttf = self.ttf_scaler.inverse_transform([[reg_out.cpu().numpy()[0]]])[0][0]
                confidence_score = confidence.cpu().numpy()[0]
            
            # Determine failure status and root cause
            failure = "Yes" if pred_class in ["Critical", "Warning"] else "No"
            cause, solution = self.analyze_ventilator_failure_cause(input_data)
            
            # Format probability text
            prob_text = "\n".join([f"{cls}: {probs[i]*100:.1f}%" for i, cls in enumerate(self.le.classes_)])
            
            return {
                'condition': pred_class,
                'failure': failure,
                'cause': cause,
                'solution': solution,
                'time_to_failure': float(pred_ttf),
                'probabilities': prob_text,
                'confidence': float(confidence_score * 100)
            }
        except Exception as e:
            print(f"Error in Ventilator prediction: {e}")
            return None

    def analyze_ventilator_failure_cause(self, input_data):
        """Analyze the root cause of potential Ventilator failure"""
        cause = "System Degradation"
        solution = "Schedule comprehensive inspection"
        
        if input_data["Filter_Status_pct"] < 30:
            cause = "Clogged Air Filter"
            solution = "Replace filter immediately"
        elif input_data["Airway_Pressure_cmH2O"] > 45 or input_data["Valve_Response_ms"] > 150:
            cause = "Airway Obstruction or Valve Delay"
            solution = "Check tubing, valves and airway patency"
        elif input_data["Oxygen_Concentration_pct"] < 40:
            cause = "Oxygen Supply Issue"
            solution = "Inspect O2 supply and mixer system"
        elif input_data["Exhaled_CO2_mmHg"] > 55:
            cause = "Ventilation Inefficiency"
            solution = "Recalibrate ventilator settings"
        elif input_data["Internal_Temperature_C"] > 60:
            cause = "Overheating"
            solution = "Check cooling system and ventilation"
        elif input_data["Battery_Level_pct"] < 20:
            cause = "Battery Failure Risk"
            solution = "Replace or recharge backup battery"
        elif input_data["Alarm_Frequency_count"] > 20:
            cause = "Frequent Alarms"
            solution = "Investigate alarm logs and root causes"
        
        return cause, solution

# Initialize all predictors
eeg_predictor = EEGFailurePredictor()
mri_predictor = MRIFailurePredictor()
ventilator_predictor = VentilatorFailurePredictor()

# Load all models
eeg_predictor.load_model_and_preprocessors()
mri_predictor.load_model_and_preprocessors()
ventilator_predictor.load_model_and_preprocessors()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Multi-Device AI Prediction API is running'})


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get device type
        device_type = data.get('device_type', 'EEG Machine')
        
        # Route to appropriate predictor based on device type
        if device_type == 'EEG Machine':
            # Validate EEG input data
            required_fields = ["temperature", "voltage", "current", "vibration", "pressure", "humidity", "usageHours"]
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing field: {field}'}), 400
            
            # Convert to the format expected by the EEG model
            input_data = {
                "Temperature(°C)": float(data["temperature"]),
                "Voltage(V)": float(data["voltage"]),
                "Current(A)": float(data["current"]),
                "Vibration(g)": float(data["vibration"]),
                "Pressure(kPa)": float(data["pressure"]),
                "Humidity(%)": float(data["humidity"]),
                "Usage Hours(h)": float(data["usageHours"])
            }
            
            result = eeg_predictor.predict_from_dict(input_data)
            
        elif device_type == 'MRI Scanner':
            # Validate MRI input data
            required_fields = [
                "helium_level", "magnetic_field", "rf_power", "gradient_temp",
                "compressor_pressure", "chiller_flow", "room_humidity", 
                "system_runtime", "vibration", "coil_temperature", 
                "power_consumption", "magnet_quench_risk"
            ]
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing field: {field}'}), 400
            
            # Convert to the format expected by the MRI model
            input_data = {
                "Helium_Level_pct": float(data["helium_level"]),
                "Magnetic_Field_T": float(data["magnetic_field"]),
                "RF_Power_kW": float(data["rf_power"]),
                "Gradient_Temp_C": float(data["gradient_temp"]),
                "Compressor_Pressure_bar": float(data["compressor_pressure"]),
                "Chiller_Flow_lpm": float(data["chiller_flow"]),
                "Room_Humidity_pct": float(data["room_humidity"]),
                "System_Runtime_hrs": float(data["system_runtime"]),
                "Vibration_mm_s": float(data["vibration"]),
                "Coil_Temperature_C": float(data["coil_temperature"]),
                "Power_Consumption_kW": float(data["power_consumption"]),
                "Magnet_Quench_Risk": float(data["magnet_quench_risk"])
            }
            
            result = mri_predictor.predict_from_dict(input_data)
            
        elif device_type == 'Ventilator':
            # Validate Ventilator input data
            required_fields = [
                "airway_pressure", "tidal_volume", "respiratory_rate",
                "oxygen_concentration", "inspiratory_flow", "exhaled_co2",
                "humidifier_temperature", "compressor_pressure", "valve_response",
                "battery_level", "power_consumption", "alarm_frequency",
                "system_runtime", "vibration", "internal_temperature", "filter_status"
            ]
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing field: {field}'}), 400
            
            # Convert to the format expected by the Ventilator model
            input_data = {
                "Airway_Pressure_cmH2O": float(data["airway_pressure"]),
                "Tidal_Volume_ml": float(data["tidal_volume"]),
                "Respiratory_Rate_bpm": float(data["respiratory_rate"]),
                "Oxygen_Concentration_pct": float(data["oxygen_concentration"]),
                "Inspiratory_Flow_lpm": float(data["inspiratory_flow"]),
                "Exhaled_CO2_mmHg": float(data["exhaled_co2"]),
                "Humidifier_Temperature_C": float(data["humidifier_temperature"]),
                "Compressor_Pressure_bar": float(data["compressor_pressure"]),
                "Valve_Response_ms": float(data["valve_response"]),
                "Battery_Level_pct": float(data["battery_level"]),
                "Power_Consumption_kW": float(data["power_consumption"]),
                "Alarm_Frequency_count": float(data["alarm_frequency"]),
                "System_Runtime_hrs": float(data["system_runtime"]),
                "Vibration_mm_s": float(data["vibration"]),
                "Internal_Temperature_C": float(data["internal_temperature"]),
                "Filter_Status_pct": float(data["filter_status"])
            }
            
            result = ventilator_predictor.predict_from_dict(input_data)
            
        else:
            return jsonify({'error': f'Unsupported device type: {device_type}'}), 400
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-prediction', methods=['POST'])
def save_prediction():
    try:
        data = request.get_json()
        
        # Validate required fields for new eeg_predictions table
        required_fields = [
            "device_id", "maintenance_log_id", "uploaded_by",
            "temperature", "voltage", "current", "vibration", "pressure", "humidity", "usage_hours",
            "predicted_condition", "predicted_failure", "predicted_cause", "recommended_solution",
            "time_to_failure_days", "confidence_score", "class_probabilities"
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Parse class probabilities from string to JSON
        if isinstance(data["class_probabilities"], str):
            # Convert string format to JSON
            prob_lines = data["class_probabilities"].strip().split('\n')
            probabilities = {}
            for line in prob_lines:
                if ':' in line:
                    class_name, percentage = line.split(':')
                    probabilities[class_name.strip()] = float(percentage.strip().replace('%', ''))
            data["class_probabilities"] = probabilities
        
        # Prepare data for database insertion (new table structure)
        prediction_data = {
            "device_id": data["device_id"],
            "maintenance_log_id": data["maintenance_log_id"],
            "uploaded_by": data["uploaded_by"],
            "temperature": float(data["temperature"]),
            "voltage": float(data["voltage"]),
            "current": float(data["current"]),
            "vibration": float(data["vibration"]),
            "pressure": float(data["pressure"]),
            "humidity": float(data["humidity"]),
            "usage_hours": float(data["usage_hours"]),
            "predicted_condition": data["predicted_condition"],
            "predicted_failure": data["predicted_failure"],
            "predicted_cause": data["predicted_cause"],
            "recommended_solution": data["recommended_solution"],
            "time_to_failure_days": float(data["time_to_failure_days"]),
            "confidence_score": float(data["confidence_score"]),
            "class_probabilities": data["class_probabilities"]
        }
        
        # Insert into eeg_predictions table
        result = supabase.table('eeg_predictions').insert(prediction_data).execute()
        
        if result.data:
            return jsonify({
                'success': True,
                'message': 'EEG prediction saved successfully',
                'prediction_id': result.data[0]['id']
            })
        else:
            return jsonify({'error': 'Failed to save prediction to database'}), 500
        
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-device', methods=['DELETE'])
def delete_device():
    """Delete a device and all its related data from Supabase"""
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        user_id = data.get('user_id')
        
        if not device_id or not user_id:
            return jsonify({'error': 'device_id and user_id are required'}), 400
        
        # Verify the user has permission to delete this device
        # First check if the device belongs to the user's hospital
        device_response = supabase.table('devices').select('hospital_id').eq('id', device_id).execute()
        
        if not device_response.data:
            return jsonify({'error': 'Device not found'}), 404
        
        # Check if user belongs to the same hospital
        user_response = supabase.table('users').select('hospital_id').eq('id', user_id).execute()
        
        if not user_response.data:
            return jsonify({'error': 'User not found'}), 404
        
        if device_response.data[0]['hospital_id'] != user_response.data[0]['hospital_id']:
            return jsonify({'error': 'Unauthorized to delete this device'}), 403
        
        # Delete related data in the correct order (due to foreign key constraints)
        
        # 1. Delete EEG predictions
        supabase.table('eeg_predictions').delete().eq('device_id', device_id).execute()
        
        # 2. Delete MRI predictions
        supabase.table('mri_predictions').delete().eq('device_id', device_id).execute()
        
        # 3. Delete Ventilator predictions
        supabase.table('ventilator_predictions').delete().eq('device_id', device_id).execute()
        
        # 4. Delete maintenance logs
        supabase.table('maintenance_logs').delete().eq('device_id', device_id).execute()
        
        # 5. Finally delete the device itself
        device_delete_response = supabase.table('devices').delete().eq('id', device_id).execute()
        
        if not device_delete_response.data:
            return jsonify({'error': 'Failed to delete device'}), 500
        
        print(f"Successfully deleted device {device_id} and all related data")
        return jsonify({'message': 'Device and all related data deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting device: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)