import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# CONFIG
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SEQ_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "lstm_ventilator_failure_model.pth"
SCALER_PATH = "ventilator_feature_scaler.pkl"
TTF_SCALER_PATH = "ventilator_ttf_scaler.pkl"
LABEL_ENCODER_PATH = "ventilator_label_encoder.pkl"
DATA_PATH = "synthetic_ventilator_dataset.csv"

# -----------------------
# VENTILATOR LSTM MODEL
# -----------------------
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
# VENTILATOR FAILURE PREDICTOR CLASS
# -----------------------
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
        self.background_samples = None
        
        # Ventilator-specific sensor ranges
        self.ranges = {
            "Airway_Pressure_cmH2O": (5, 60),
            "Tidal_Volume_ml": (200, 900),
            "Respiratory_Rate_bpm": (8, 40),
            "Oxygen_Concentration_pct": (21, 100),
            "Inspiratory_Flow_lpm": (10, 120),
            "Exhaled_CO2_mmHg": (20, 60),
            "Humidifier_Temperature_C": (25, 45),
            "Compressor_Pressure_bar": (1, 8),
            "Valve_Response_ms": (10, 200),
            "Battery_Level_pct": (0, 100),
            "Power_Consumption_kW": (0.2, 2.0),
            "Alarm_Frequency_count": (0, 50),
            "System_Runtime_hrs": (0, 20000),
            "Vibration_mm_s": (0.1, 5.0),
            "Internal_Temperature_C": (20, 70),
            "Filter_Status_pct": (0, 100),
        }

    # -----------------------
    # Load model + preprocessors
    # -----------------------
    def load_model_and_preprocessors(self):
        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.ttf_scaler = joblib.load(TTF_SCALER_PATH)
            self.le = joblib.load(LABEL_ENCODER_PATH)
            
            self.model = VentilatorMultiTaskLSTM(
                n_features=len(self.feature_names), 
                n_classes=len(self.le.classes_)
            )
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()
            
            df = pd.read_csv(DATA_PATH)
            self.background_samples = df[self.feature_names].values[:500]
            
            print("✅ Ventilator Model and preprocessors loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Ventilator model: {e}")
            return False

    # -----------------------
    # Normal Ranges
    # -----------------------
    def get_normal_range(self, sensor):
        normal_ranges = {
            "Airway_Pressure_cmH2O": "20-30 cmH2O",
            "Tidal_Volume_ml": "400-600 ml",
            "Respiratory_Rate_bpm": "12-20 bpm",
            "Oxygen_Concentration_pct": "30-60%",
            "Inspiratory_Flow_lpm": "40-80 L/min",
            "Exhaled_CO2_mmHg": "35-45 mmHg",
            "Humidifier_Temperature_C": "34-37 °C",
            "Compressor_Pressure_bar": "3-6 bar",
            "Valve_Response_ms": "20-80 ms",
            "Battery_Level_pct": "50-100%",
            "Power_Consumption_kW": "0.5-1.5 kW",
            "Alarm_Frequency_count": "0-5",
            "System_Runtime_hrs": "0-10000 hrs",
            "Vibration_mm_s": "0.5-2.0 mm/s",
            "Internal_Temperature_C": "30-50 °C",
            "Filter_Status_pct": "60-100%"
        }
        return normal_ranges.get(sensor, "N/A")

    # -----------------------
    # Get user input
    # -----------------------
    def get_user_input(self):
        print("\n🔮 Enter Ventilator machine sensor readings:")
        input_data = {}
        
        for feat in self.feature_names:
            low, high = self.ranges[feat]
            while True:
                try:
                    unit = self._get_unit(feat)
                    normal_range = self.get_normal_range(feat)
                    
                    prompt = f"Enter {feat.replace('_', ' ')} ({low}-{high} {unit}, normal: {normal_range}): "
                    val = float(input(prompt))
                    
                    if low <= val <= high:
                        input_data[feat] = val
                        break
                    else:
                        print(f"⚠️ Value must be between {low} and {high} {unit}. Try again.")
                        
                except ValueError:
                    print("⚠️ Invalid input. Please enter a number.")
        
        return input_data

    def _get_unit(self, feat):
        if "pct" in feat or "Level" in feat or "Status" in feat:
            return "%"
        elif "Pressure" in feat:
            return "cmH2O" if "Airway" in feat else "bar"
        elif "Volume" in feat:
            return "ml"
        elif "Rate" in feat:
            return "bpm"
        elif "Flow" in feat:
            return "L/min"
        elif "CO2" in feat:
            return "mmHg"
        elif "Temperature" in feat:
            return "°C"
        elif "ms" in feat:
            return "ms"
        elif "hrs" in feat:
            return "hours"
        elif "mm_s" in feat:
            return "mm/s"
        elif "kW" in feat:
            return "kW"
        elif "count" in feat:
            return "events"
        else:
            return ""

    # -----------------------
    # Prediction
    # -----------------------
    def predict_from_dict(self, input_data):
        X = self.scaler.transform(pd.DataFrame([input_data], columns=self.feature_names))
        seq = np.tile(X[:, None, :], (1, SEQ_LEN, 1))
        seq_t = torch.tensor(seq, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            cls_logits, reg_out, confidence = self.model(seq_t)
            probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
            pred_class = self.le.inverse_transform([np.argmax(probs)])[0]
            pred_ttf = self.ttf_scaler.inverse_transform([[reg_out.cpu().numpy()[0]]])[0][0]
            confidence_score = confidence.cpu().numpy()[0]
        
        failure = "Yes" if pred_class in ["Critical", "Warning"] else "No"
        cause, solution = self.analyze_ventilator_failure_cause(input_data)
        
        prob_text = "\n".join([f"{cls}: {probs[i]*100:.1f}%" for i, cls in enumerate(self.le.classes_)])
        
        if confidence_score > 0.8:
            risk_level = "High Confidence"
        elif confidence_score > 0.6:
            risk_level = "Medium Confidence"
        else:
            risk_level = "Low Confidence - Manual Review Recommended"
        
        return {
            'condition': pred_class,
            'failure': failure,
            'cause': cause,
            'solution': solution,
            'time_to_failure': pred_ttf,
            'probabilities': prob_text,
            'confidence': confidence_score,
            'risk_level': risk_level,
            'raw_input': X
        }

    # -----------------------
    # Failure cause analysis
    # -----------------------
    def analyze_ventilator_failure_cause(self, input_data):
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

    # -----------------------
    # Explainability
    # -----------------------
    def explain_prediction(self, X):
        print("\n🔍 Ventilator Prediction Explainability Analysis")
        
        try:
            background = self.background_samples[:50]
            
            def model_predict(x):
                x_tensor = torch.tensor(np.tile(x[:, None, :], (1, SEQ_LEN, 1))).float().to(DEVICE)
                with torch.no_grad():
                    cls_logits, _, _ = self.model(x_tensor)
                    return torch.softmax(cls_logits, dim=1).cpu().numpy()
            
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X, nsamples=50)
            
            top_shap = np.argsort(np.abs(shap_values[0][0]))[::-1][:3]
            
            print("\n🎯 Top SHAP features influencing prediction:")
            for i in top_shap:
                impact = shap_values[0][0][i]
                direction = "↑ Increases" if impact > 0 else "↓ Decreases"
                print(f"  {self.feature_names[i]}: {direction} failure risk (impact: {impact:.4f})")
            
            lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=self.scaler.transform(self.background_samples[:200]),
                feature_names=self.feature_names,
                class_names=self.le.classes_,
                discretize_continuous=True
            )
            
            lime_exp = lime_explainer.explain_instance(
                X[0], model_predict, num_features=len(self.feature_names)
            )
            
            print("\n🔍 LIME explanation (top contributing features):")
            for feature, weight in lime_exp.as_list()[:3]:
                direction = "Risk Factor" if weight > 0 else "Protective Factor"
                print(f"  {feature}: {direction} (weight: {weight:.4f})")
                
        except Exception as e:
            print(f"⚠️ Explainability analysis failed: {e}")

    # -----------------------
    # Input correlation validation
    # -----------------------
    def validate_input_correlations(self, input_data):
        warnings = []
        
        if input_data["Airway_Pressure_cmH2O"] > 40 and input_data["Tidal_Volume_ml"] < 300:
            warnings.append("⚠️ High airway pressure but very low tidal volume")
        
        if input_data["Oxygen_Concentration_pct"] > 90 and input_data["Exhaled_CO2_mmHg"] > 50:
            warnings.append("⚠️ High O2 concentration but poor CO2 clearance")
        
        if input_data["System_Runtime_hrs"] > 15000 and input_data["Vibration_mm_s"] < 1.0:
            warnings.append("⚠️ Low vibration unexpected for high runtime")
        
        return warnings

    # -----------------------
    # Interactive loop
    # -----------------------
    def interactive_prediction(self):
        while True:
            try:
                print("\n" + "="*60)
                print("🏥 VENTILATOR FAILURE PREDICTION")
                print("="*60)
                
                input_data = self.get_user_input()
                
                warnings = self.validate_input_correlations(input_data)
                if warnings:
                    print("\n⚠️ Input Validation Warnings:")
                    for warning in warnings:
                        print(f"  {warning}")
                    
                    proceed = input("\n❓ Continue with prediction? (y/n): ").lower()
                    if proceed != 'y':
                        continue
                
                result = self.predict_from_dict(input_data)
                
                print("\n" + "="*50)
                print("📊 VENTILATOR FAILURE PREDICTION RESULTS")
                print("="*50)
                print(f"🔍 Condition: {result['condition']}")
                print(f"⚠️ Failure Risk: {result['failure']}")
                print(f"🎯 Root Cause: {result['cause']}")
                print(f"🔧 Recommended Action: {result['solution']}")
                print(f"📅 Estimated Time-to-Failure: {result['time_to_failure']:.0f} days")
                print(f"🎯 Confidence: {result['confidence']:.1%} ({result['risk_level']})")
                
                print(f"\n📊 Condition Probabilities:")
                print(result['probabilities'])
                
                self.explain_prediction(result['raw_input'])
                
                print("\n" + "="*50)
                another = input("🔄 Analyze another ventilator system? (y/n): ").lower()
                if another != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Prediction stopped by user")
                break
            except Exception as e:
                print(f"❌ Error during prediction: {e}")

# -----------------------
# MAIN EXECUTION
# -----------------------
def main():
    print("🏥 VENTILATOR FAILURE PREDICTION SYSTEM")
    print("=" * 55)
    print("🔬 Advanced Multi-Modal Analysis with Confidence Scoring")
    print("🧠 Enhanced LSTM with Attention Mechanism")
    print("⚡ Real-time Correlation Validation")
    print("=" * 55)
    
    predictor = VentilatorFailurePredictor()
    
    if not predictor.load_model_and_preprocessors():
        print("❌ Failed to load ventilator model. Please ensure all model files exist.")
        print("📁 Required files:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {SCALER_PATH}")
        print(f"  - {TTF_SCALER_PATH}")
        print(f"  - {LABEL_ENCODER_PATH}")
        return
    
    predictor.interactive_prediction()
    
    print("\n🏥 Thank you for using the Ventilator Failure Prediction System!")
    print("💡 Remember to validate predictions with qualified technicians")

if __name__ == "__main__":
    main()
