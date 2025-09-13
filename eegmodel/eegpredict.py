import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import shap
from lime import lime_tabular

# -----------------------
# CONFIG
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SEQ_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "lstm_eeg_failure_model.pth"
SCALER_PATH = "feature_scaler.pkl"
TTF_SCALER_PATH = "ttf_scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
DATA_PATH = "synthetic_eeg_machine_dataset.csv"  # For background samples

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
        self.background_samples = None
        
        self.ranges = {
            "Temperature(°C)": (30, 80),
            "Voltage(V)": (180, 250),
            "Current(A)": (0, 15),
            "Vibration(g)": (0, 5),
            "Pressure(kPa)": (80, 120),
            "Humidity(%)": (20, 80),
            "Usage Hours(h)": (0, 10000),
        }
    
    def load_model_and_preprocessors(self):
        """Load the trained model and all preprocessors"""
        try:
            # Load preprocessors
            self.scaler = joblib.load(SCALER_PATH)
            self.ttf_scaler = joblib.load(TTF_SCALER_PATH)
            self.le = joblib.load(LABEL_ENCODER_PATH)
            
            # Initialize and load model
            self.model = MultiTaskLSTM(n_features=len(self.feature_names), n_classes=len(self.le.classes_))
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()
            
            # Load background samples for explainability
            df = pd.read_csv(DATA_PATH)
            self.background_samples = df[self.feature_names].values[:500]
            
            print("✅ Model and preprocessors loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_user_input(self):
        """Get sensor readings from user input"""
        print("\n🔮 Enter new machine sensor readings:")
        input_data = {}
        
        for feat in self.feature_names:
            low, high = self.ranges[feat]
            while True:
                try:
                    val = float(input(f"Enter {feat} ({low} to {high}): "))
                    if low <= val <= high:
                        input_data[feat] = val
                        break
                    else:
                        print(f"⚠️ Value must be between {low} and {high}. Try again.")
                except ValueError:
                    print("⚠️ Invalid input. Please enter a number.")
        
        return input_data
    
    def predict_from_dict(self, input_data):
        """Make prediction from input dictionary"""
        # Scale input
        X = self.scaler.transform(pd.DataFrame([input_data], columns=self.feature_names))
        seq = np.tile(X[:, None, :], (1, SEQ_LEN, 1))
        seq_t = torch.tensor(seq, dtype=torch.float32).to(DEVICE)
        
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
            'time_to_failure': pred_ttf,
            'probabilities': prob_text,
            'raw_input': X
        }
    
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
    
    def explain_prediction(self, X):
        """Generate SHAP and LIME explanations"""
        print("\n🔍 Explainability Analysis")
        
        try:
            # SHAP explanation
            background = self.background_samples[:50]
            explainer = shap.KernelExplainer(
                lambda X: torch.softmax(
                    self.model(torch.tensor(np.tile(X[:, None, :], (1, SEQ_LEN, 1))).float().to(DEVICE))[0], dim=1
                ).cpu().detach().numpy(),
                background
            )
            shap_values = explainer.shap_values(X, nsamples=50)
            top_shap = np.argsort(np.abs(shap_values[0][0]))[::-1][:3]
            print("\nTop SHAP features influencing prediction:")
            for i in top_shap:
                print(f"- {self.feature_names[i]} (impact: {shap_values[0][0][i]:.4f})")
            
            # LIME explanation
            lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=self.scaler.transform(self.background_samples[:200]),
                feature_names=self.feature_names,
                class_names=self.le.classes_,
                discretize_continuous=True
            )
            lime_exp = lime_explainer.explain_instance(
                X[0],
                lambda x: torch.softmax(
                    self.model(torch.tensor(np.tile(x[:, None, :], (1, SEQ_LEN, 1))).float().to(DEVICE))[0], dim=1
                ).cpu().detach().numpy()
            )
            print("\nLIME explanation (top features):")
            for f in lime_exp.as_list()[:3]:
                print(f"- {f[0]} (weight: {f[1]:.4f})")
                
        except Exception as e:
            print(f"⚠️ Explainability analysis failed: {e}")
    
    def interactive_prediction(self):
        """Interactive prediction loop"""
        while True:
            try:
                # Get user input
                input_data = self.get_user_input()
                
                # Make prediction
                result = self.predict_from_dict(input_data)
                
                # Display results
                print("\n✅ Prediction Result")
                print(f"Condition: {result['condition']}")
                print(f"Failure: {result['failure']}")
                print(f"Cause of Failure: {result['cause']}")
                print(f"Solution: {result['solution']}")
                print(f"Predicted Time-to-Failure: {result['time_to_failure']:.1f} days")
                print(f"\n📊 Class Probabilities:\n{result['probabilities']}")
                
                # Generate explanations
                self.explain_prediction(result['raw_input'])
                
                # Ask for another prediction
                another = input("\n🔄 Make another prediction? (y/n): ").lower()
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
    print("🚀 EEG Machine Failure Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EEGFailurePredictor()
    
    # Load model and preprocessors
    if not predictor.load_model_and_preprocessors():
        print("❌ Failed to load model. Please ensure all model files exist.")
        return
    
    # Start interactive prediction
    predictor.interactive_prediction()
    
    print("\n👋 Thank you for using the EEG Failure Prediction System!")

if __name__ == "__main__":
    main()
