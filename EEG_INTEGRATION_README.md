# EEG Machine Failure Prediction Integration

## Overview

This document describes the integration of the EEG machine failure prediction model into the MendWorks hospital equipment maintenance system.

## Features Added

### 1. EEG Maintenance Log Page
- **Location**: `/device/:deviceId/eeg-maintenance`
- **Purpose**: Input sensor readings for AI-powered failure prediction
- **Features**:
  - 7 sensor input fields with validation
  - Real-time prediction using LSTM model
  - Visual condition indicators
  - Root cause analysis
  - Recommended solutions
  - Time-to-failure prediction

### 2. Python Backend API
- **Location**: `backend/app.py`
- **Port**: 5000
- **Endpoints**:
  - `GET /api/health` - Health check
  - `POST /api/predict` - EEG failure prediction

### 3. Dashboard Integration
- **EEG-specific button**: "AI Analysis" instead of "Add Log"
- **Purple theme**: Matches EEG machine branding
- **Brain icon**: Visual indicator for EEG devices

## Sensor Inputs

The system accepts 7 sensor readings:

| Sensor | Range | Unit | Description |
|--------|-------|------|-------------|
| Temperature | 30-80 | °C | Machine operating temperature |
| Voltage | 180-250 | V | Electrical supply voltage |
| Current | 0-15 | A | Electrical current draw |
| Vibration | 0-5 | g | Mechanical vibration level |
| Pressure | 80-120 | kPa | System pressure |
| Humidity | 20-80 | % | Environmental humidity |
| Usage Hours | 0-10,000 | h | Total operating hours |

## Model Outputs

### Condition Classes
- **Good**: Normal operation, no issues
- **Average**: Minor wear, routine maintenance needed
- **Warning**: Potential issues, inspection recommended
- **Critical**: Immediate attention required

### Additional Information
- **Failure Prediction**: Yes/No
- **Time to Failure**: Days until predicted failure
- **Root Cause**: Specific failure cause analysis
- **Solution**: Recommended maintenance action
- **Confidence**: Prediction confidence percentage

## Setup Instructions

### 1. Start the Python Backend
```bash
# Make the script executable
chmod +x start_backend.sh

# Start the backend
./start_backend.sh
```

### 2. Start the React Frontend
```bash
npm run dev
```

### 3. Access the Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000

## File Structure

```
project/
├── src/pages/
│   └── EEGMaintenanceLog.tsx    # EEG maintenance log page
├── backend/
│   ├── app.py                   # Flask API server
│   └── requirements.txt         # Python dependencies
├── eegmodel/
│   ├── eegpredict.py           # Original prediction script
│   ├── lstm_eeg_failure_model.pth  # Trained LSTM model
│   ├── feature_scaler.pkl      # Feature normalization
│   ├── ttf_scaler.pkl          # Time-to-failure scaling
│   ├── label_encoder.pkl       # Class label encoding
│   └── synthetic_eeg_machine_dataset.csv  # Training data
└── start_backend.sh            # Backend startup script
```

## API Usage

### Prediction Request
```javascript
POST http://localhost:5000/api/predict
Content-Type: application/json

{
  "temperature": 45.5,
  "voltage": 220.0,
  "current": 7.2,
  "vibration": 1.5,
  "pressure": 95.0,
  "humidity": 50.0,
  "usageHours": 5000
}
```

### Prediction Response
```javascript
{
  "condition": "Good",
  "failure": "No",
  "cause": "None",
  "solution": "Continue routine maintenance",
  "time_to_failure": 365.0,
  "probabilities": "Good: 60.0%\nAverage: 25.0%\nWarning: 10.0%\nCritical: 5.0%",
  "confidence": 85.2
}
```

## Error Handling

- **API Unavailable**: Falls back to simulation mode
- **Invalid Inputs**: Client-side validation with error messages
- **Model Errors**: Graceful error handling with user feedback

## Future Enhancements

1. **Database Integration**: Store sensor readings and predictions
2. **Historical Analysis**: Track prediction accuracy over time
3. **Alert System**: Notify technicians of critical predictions
4. **Batch Processing**: Analyze multiple devices simultaneously
5. **Model Retraining**: Update model with new data

## Troubleshooting

### Backend Issues
- Ensure Python 3.8+ is installed
- Check that all model files exist in `eegmodel/` directory
- Verify virtual environment is activated
- Check port 5000 is available

### Frontend Issues
- Ensure React dev server is running on port 5173
- Check browser console for API connection errors
- Verify EEG device is selected in dashboard

## Model Performance

The LSTM model provides:
- **Multi-task learning**: Classification + regression
- **Sequence modeling**: 30-timestep input sequences
- **Explainability**: SHAP and LIME integration
- **Real-time prediction**: < 100ms response time

## Security Considerations

- API runs on localhost only
- No authentication required for development
- Model files should be secured in production
- Consider rate limiting for production use
