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

MODEL_PATH = "lstm_mri_failure_model.pth"
SCALER_PATH = "mri_feature_scaler.pkl"
TTF_SCALER_PATH = "mri_ttf_scaler.pkl"
LABEL_ENCODER_PATH = "mri_label_encoder.pkl"
DATA_PATH = "synthetic_mri_machine_dataset.csv"

# -----------------------
# MRI LSTM MODEL (Same as training)
# -----------------------
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

# -----------------------
# MRI FAILURE PREDICTOR CLASS
# -----------------------
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
        self.background_samples = None
        
        # MRI-specific sensor ranges
        self.ranges = {
            "Helium_Level_pct": (60, 95),
            "Magnetic_Field_T": (2.5, 3.1),
            "RF_Power_kW": (20, 60),
            "Gradient_Temp_C": (15, 55),
            "Compressor_Pressure_bar": (3, 20),
            "Chiller_Flow_lpm": (15, 70),
            "Room_Humidity_pct": (25, 90),
            "System_Runtime_hrs": (0, 15000),
            "Vibration_mm_s": (0.1, 8.0),
            "Coil_Temperature_C": (18, 55),
            "Power_Consumption_kW": (40, 95),
            "Magnet_Quench_Risk": (0.01, 0.6)
        }
    
    def load_model_and_preprocessors(self):
        """Load the trained MRI model and all preprocessors"""
        try:
            # Load preprocessors
            self.scaler = joblib.load(SCALER_PATH)
            self.ttf_scaler = joblib.load(TTF_SCALER_PATH)
            self.le = joblib.load(LABEL_ENCODER_PATH)
            
            # Initialize and load model
            self.model = MRIMultiTaskLSTM(n_features=len(self.feature_names), n_classes=len(self.le.classes_))
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()
            
            # Load background samples for explainability
            df = pd.read_csv(DATA_PATH)
            self.background_samples = df[self.feature_names].values[:500]
            
            print("✅ MRI Model and preprocessors loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading MRI model: {e}")
            return False
    
    def get_user_input(self):
        """Get MRI sensor readings from user input with validation"""
        print("\n🔮 Enter MRI machine sensor readings:")
        input_data = {}
        
        for feat in self.feature_names:
            low, high = self.ranges[feat]
            while True:
                try:
                    # Display units and normal ranges
                    if "pct" in feat:
                        unit = "%"
                        normal_range = self.get_normal_range(feat)
                    elif "T" in feat:
                        unit = "Tesla"
                        normal_range = self.get_normal_range(feat)
                    elif "kW" in feat:
                        unit = "kW"
                        normal_range = self.get_normal_range(feat)
                    elif "C" in feat:
                        unit = "°C"
                        normal_range = self.get_normal_range(feat)
                    elif "bar" in feat:
                        unit = "bar"
                        normal_range = self.get_normal_range(feat)
                    elif "lpm" in feat:
                        unit = "L/min"
                        normal_range = self.get_normal_range(feat)
                    elif "hrs" in feat:
                        unit = "hours"
                        normal_range = self.get_normal_range(feat)
                    elif "mm_s" in feat:
                        unit = "mm/s"
                        normal_range = self.get_normal_range(feat)
                    else:
                        unit = ""
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
    
    def get_normal_range(self, sensor):
        """Get normal operating range for a sensor"""
        normal_ranges = {
            "Helium_Level_pct": "87-92%",
            "Magnetic_Field_T": "2.98-3.02T",
            "RF_Power_kW": "27-32kW",
            "Gradient_Temp_C": "20-25°C",
            "Compressor_Pressure_bar": "14-17 bar",
            "Chiller_Flow_lpm": "50-60 L/min",
            "Room_Humidity_pct": "45-55%",
            "System_Runtime_hrs": "0-5000 hrs",
            "Vibration_mm_s": "0.5-1.2 mm/s",
            "Coil_Temperature_C": "22-28°C",
            "Power_Consumption_kW": "48-58 kW",
            "Magnet_Quench_Risk": "0.01-0.05"
        }
        return normal_ranges.get(sensor, "N/A")
    
    def predict_from_dict(self, input_data):
        """Make prediction from input dictionary with confidence scoring"""
        # Scale input
        X = self.scaler.transform(pd.DataFrame([input_data], columns=self.feature_names))
        seq = np.tile(X[:, None, :], (1, SEQ_LEN, 1))
        seq_t = torch.tensor(seq, dtype=torch.float32).to(DEVICE)
        
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
        
        # Risk assessment based on confidence
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
    
    def analyze_mri_failure_cause(self, input_data):
        """Analyze the root cause of potential MRI failure"""
        cause = "System Degradation"
        solution = "Schedule comprehensive inspection"
        
        # MRI-specific failure analysis
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
    
    def explain_prediction(self, X):
        """Generate SHAP and LIME explanations for MRI predictions"""
        print("\n🔍 MRI Prediction Explainability Analysis")
        
        try:
            # SHAP explanation
            background = self.background_samples[:50]
            
            def model_predict(x):
                x_tensor = torch.tensor(np.tile(x[:, None, :], (1, SEQ_LEN, 1))).float().to(DEVICE)
                with torch.no_grad():
                    cls_logits, _, _ = self.model(x_tensor)
                    return torch.softmax(cls_logits, dim=1).cpu().numpy()
            
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X, nsamples=50)
            
            # Find top influential features
            top_shap = np.argsort(np.abs(shap_values[0][0]))[::-1][:3]
            
            print("\n🎯 Top SHAP features influencing prediction:")
            for i in top_shap:
                impact = shap_values[0][0][i]
                direction = "↑ Increases" if impact > 0 else "↓ Decreases"
                print(f"  {self.feature_names[i]}: {direction} failure risk (impact: {impact:.4f})")
            
            # LIME explanation
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
    
    def validate_input_correlations(self, input_data):
        """Validate input data for realistic MRI sensor correlations"""
        warnings = []
        
        # Check helium level vs magnetic field correlation
        expected_field = 3.0 + (input_data["Helium_Level_pct"] - 89.5) * 0.002
        if abs(input_data["Magnetic_Field_T"] - expected_field) > 0.02:
            warnings.append("⚠️ Magnetic field inconsistent with helium level")
        
        # Check RF power vs coil temperature correlation
        expected_coil_temp = 25 + (input_data["RF_Power_kW"] - 29.5) * 0.8
        if abs(input_data["Coil_Temperature_C"] - expected_coil_temp) > 5:
            warnings.append("⚠️ Coil temperature inconsistent with RF power")
        
        # Check system runtime vs vibration correlation
        if input_data["System_Runtime_hrs"] > 8000 and input_data["Vibration_mm_s"] < 2.0:
            warnings.append("⚠️ Low vibration unexpected for high runtime")
        
        return warnings
    
    def interactive_prediction(self):
        """Interactive MRI prediction loop with enhanced validation"""
        while True:
            try:
                print("\n" + "="*60)
                print("🏥 MRI MACHINE FAILURE PREDICTION")
                print("="*60)
                
                # Get user input
                input_data = self.get_user_input()
                
                # Validate correlations
                warnings = self.validate_input_correlations(input_data)
                if warnings:
                    print("\n⚠️ Input Validation Warnings:")
                    for warning in warnings:
                        print(f"  {warning}")
                    
                    proceed = input("\n❓ Continue with prediction? (y/n): ").lower()
                    if proceed != 'y':
                        continue
                
                # Make prediction
                result = self.predict_from_dict(input_data)
                
                # Display results
                print("\n" + "="*50)
                print("📊 MRI FAILURE PREDICTION RESULTS")
                print("="*50)
                print(f"🔍 Condition: {result['condition']}")
                print(f"⚠️ Failure Risk: {result['failure']}")
                print(f"🎯 Root Cause: {result['cause']}")
                print(f"🔧 Recommended Action: {result['solution']}")
                print(f"📅 Estimated Time-to-Failure: {result['time_to_failure']:.0f} days")
                print(f"🎯 Confidence: {result['confidence']:.1%} ({result['risk_level']})")
                
                print(f"\n📊 Condition Probabilities:")
                print(result['probabilities'])
                
                # Generate explanations
                self.explain_prediction(result['raw_input'])
                
                # Ask for another prediction
                print("\n" + "="*50)
                another = input("🔄 Analyze another MRI system? (y/n): ").lower()
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
    print("🏥 MRI MACHINE FAILURE PREDICTION SYSTEM")
    print("=" * 55)
    print("🔬 Advanced Multi-Modal Analysis with Confidence Scoring")
    print("🧠 Enhanced LSTM with Attention Mechanism")
    print("⚡ Real-time Correlation Validation")
    print("=" * 55)
    
    # Initialize predictor
    predictor = MRIFailurePredictor()
    
    # Load model and preprocessors
    if not predictor.load_model_and_preprocessors():
        print("❌ Failed to load MRI model. Please ensure all model files exist.")
        print("📁 Required files:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {SCALER_PATH}")
        print(f"  - {TTF_SCALER_PATH}")
        print(f"  - {LABEL_ENCODER_PATH}")
        return
    
    # Start interactive prediction
    predictor.interactive_prediction()
    
    print("\n🏥 Thank you for using the MRI Failure Prediction System!")
    print("💡 Remember to validate predictions with qualified technicians")

if __name__ == "__main__":
    main()
