import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import shap
from lime import lime_tabular
import joblib

# -----------------------
# CONFIG
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA_PATH = "synthetic_mri_machine_dataset.csv"
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "lstm_mri_failure_model.pth"
SCALER_PATH = "mri_feature_scaler.pkl"
TTF_SCALER_PATH = "mri_ttf_scaler.pkl"
LABEL_ENCODER_PATH = "mri_label_encoder.pkl"

# -----------------------
# DATASET CLASS
# -----------------------
class MRISeqDataset(Dataset):
    def __init__(self, seqs, cls_labels, reg_labels):
        self.seqs = seqs.astype(np.float32)
        self.cls = cls_labels.astype(np.int64)
        self.reg = reg_labels.astype(np.float32)
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.seqs[idx], dtype=torch.float32),
            torch.tensor(self.cls[idx], dtype=torch.long),
            torch.tensor(self.reg[idx], dtype=torch.float32),
        )

# -----------------------
# ENHANCED LSTM MODEL FOR MRI
# -----------------------
class MRIMultiTaskLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=128, n_layers=3, n_classes=4, dropout=0.3):
        super().__init__()
        
        # Enhanced LSTM for complex MRI correlations
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=n_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention mechanism for important sensor focus
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        
        # Enhanced feature processing
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim // 2)
        
        # Output heads
        self.cls_head = nn.Linear(hidden_dim // 2, n_classes)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)
        
        # False positive reduction layer
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention mechanism
        lstm_out = lstm_out.transpose(0, 1)  # (seq, batch, hidden*2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq, hidden*2)
        
        # Use last time step
        last_hidden = attn_out[:, -1, :]  # (batch, hidden*2)
        
        # Feature processing
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm(x)
        
        # Outputs
        cls_logits = self.cls_head(x)
        reg_out = self.reg_head(x).squeeze(1)
        confidence = torch.sigmoid(self.confidence_head(x)).squeeze(1)
        
        return cls_logits, reg_out, confidence

# -----------------------
# TRAINING & EVALUATION
# -----------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for x, y_cls, y_reg in train_loader:
            x, y_cls, y_reg = x.to(DEVICE), y_cls.to(DEVICE), y_reg.to(DEVICE)
            
            optimizer.zero_grad()
            cls_logits, reg_out, confidence = model(x)
            
            # Multi-task loss with confidence weighting
            loss_cls = cls_loss_fn(cls_logits, y_cls)
            loss_reg = reg_loss_fn(reg_out, y_reg)
            
            # Confidence loss (higher confidence for correct predictions)
            cls_pred = torch.argmax(cls_logits, dim=1)
            correct_preds = (cls_pred == y_cls).float()
            confidence_loss = nn.BCELoss()(confidence, correct_preds)
            
            # Combined loss
            total_loss_batch = loss_cls + 0.6 * loss_reg + 0.2 * confidence_loss
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += total_loss_batch.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_cls, y_reg in val_loader:
                x, y_cls, y_reg = x.to(DEVICE), y_cls.to(DEVICE), y_reg.to(DEVICE)
                cls_logits, reg_out, confidence = model(x)
                
                loss_cls = cls_loss_fn(cls_logits, y_cls)
                loss_reg = reg_loss_fn(reg_out, y_reg)
                val_loss += (loss_cls + 0.6 * loss_reg).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_PATH.replace('.pth', '_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH.replace('.pth', '_best.pth')))
    return model

def evaluate_model(model, loader, le, ttf_scaler):
    model.eval()
    preds_cls, trues_cls, preds_reg, trues_reg = [], [], [], []
    confidences = []
    
    with torch.no_grad():
        for x, y_cls, y_reg in loader:
            x = x.to(DEVICE)
            cls_logits, reg_out, confidence = model(x)
            
            probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
            preds_cls.extend(np.argmax(probs, axis=1))
            trues_cls.extend(y_cls.numpy())
            preds_reg.extend(reg_out.cpu().numpy())
            trues_reg.extend(y_reg.numpy())
            confidences.extend(confidence.cpu().numpy())
    
    # Transform back to original scale
    preds_reg = ttf_scaler.inverse_transform(np.array(preds_reg).reshape(-1, 1)).flatten()
    trues_reg = ttf_scaler.inverse_transform(np.array(trues_reg).reshape(-1, 1)).flatten()
    
    # Calculate metrics
    acc = accuracy_score(trues_cls, preds_cls)
    f1 = f1_score(trues_cls, preds_cls, average="weighted")
    mse = mean_squared_error(trues_reg, preds_reg)
    mae = mean_absolute_error(trues_reg, preds_reg)
    r2 = r2_score(trues_reg, preds_reg)
    
    # False positive analysis
    confidences = np.array(confidences)
    high_conf_mask = confidences > 0.7
    high_conf_acc = accuracy_score(
        np.array(trues_cls)[high_conf_mask], 
        np.array(preds_cls)[high_conf_mask]
    ) if np.sum(high_conf_mask) > 0 else 0
    
    print("\n📊 MRI Model Evaluation Metrics")
    print(f"Classification Accuracy: {acc:.3f} | F1-score: {f1:.3f}")
    print(f"Regression -> MSE: {mse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")
    print(f"High Confidence (>0.7) Accuracy: {high_conf_acc:.3f}")
    print(f"Average Confidence: {np.mean(confidences):.3f}")

# -----------------------
# MAIN FUNCTION
# -----------------------
def main():
    print("🚀 MRI Machine Failure Prediction Training")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"📊 Loaded {len(df)} samples")
    
    # Feature columns (MRI-specific sensors)
    feature_cols = [
        "Helium_Level_pct", "Magnetic_Field_T", "RF_Power_kW", "Gradient_Temp_C",
        "Compressor_Pressure_bar", "Chiller_Flow_lpm", "Room_Humidity_pct", 
        "System_Runtime_hrs", "Vibration_mm_s", "Coil_Temperature_C", 
        "Power_Consumption_kW", "Magnet_Quench_Risk"
    ]
    
    # Encode labels
    le = LabelEncoder()
    df["Condition_enc"] = le.fit_transform(df["Condition"])
    
    # Scale time-to-failure
    ttf_scaler = MinMaxScaler()
    df["TTF_scaled"] = ttf_scaler.fit_transform(df[["Time_to_Failure_days"]])
    
    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Create sequences
    seqs = np.tile(df[feature_cols].values[:, None, :], (1, SEQ_LEN, 1))
    
    # Split data
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        seqs, df["Condition_enc"].values, df["TTF_scaled"].values, 
        test_size=0.2, stratify=df["Condition_enc"], random_state=SEED
    )
    
    # Create data loaders
    train_dataset = MRISeqDataset(X_train, y_cls_train, y_reg_train)
    test_dataset = MRISeqDataset(X_test, y_cls_test, y_reg_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    model = MRIMultiTaskLSTM(n_features=len(feature_cols), n_classes=len(le.classes_))
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("\n🏋️ Starting training...")
    model = train_model(model, train_loader, test_loader)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    evaluate_model(model, test_loader, le, ttf_scaler)
    
    # Save artifacts
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(ttf_scaler, TTF_SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    
    print(f"\n💾 Model and preprocessors saved successfully!")
    print(f"📁 Model: {MODEL_PATH}")
    print(f"📁 Scaler: {SCALER_PATH}")
    print(f"📁 TTF Scaler: {TTF_SCALER_PATH}")
    print(f"📁 Label Encoder: {LABEL_ENCODER_PATH}")

if __name__ == "__main__":
    main()
