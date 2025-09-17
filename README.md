## MendWorks

An end-to-end web app for hospital device monitoring with AI-assisted failure prediction. Built with React + TypeScript (Vite) on the frontend, Flask on the backend, and Supabase for auth, database, and Row Level Security (RLS).

### Features
- **Device management**: Add/setup devices (EEG, MRI, Ventilator) per hospital
- **AI predictions**: LSTM models infer condition and time-to-failure (TTF)
- **Maintenance logs**: Save readings and model outputs to Supabase
- **Dashboard**: Status overview, color-coded cards, critical alerts popup
- **Auth & RLS**: Supabase authentication with per-hospital row-level access
- **Device deletion**: Backend endpoint removes a device and related data

### Tech Stack
- Frontend: React, TypeScript, Vite, Tailwind, react-router-dom, lucide-react
- Backend: Flask (Python), PyTorch (for model inference)
- Database: Supabase Postgres + RLS

### Repository Layout (high level)
- `src/` – frontend app (pages, components, auth context)
- `backend/` – Flask API (`app.py`), Python venv (ignored)
- `supabase/migrations/` – SQL migrations (devices, predictions, etc.)
- `allmodels/` – model artifacts (`.pth` weights, `.pkl` scalers/encoders)
- `eegmodel/` – EEG model artifacts

### Prerequisites
- Node.js 18+
- Python 3.11+
- A Supabase project with URL and anon key

### Environment Variables
Create `.env` files.

Frontend (`.env` at repo root):
```
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

Backend (`backend/.env` or system env):
```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Install & Run (Windows PowerShell)
Frontend:
```
cd "C:\Cursor Projects\MendWorks\MendWorks"
npm install
npm run dev
```

Backend (global Python install):
```
cd "C:\Cursor Projects\MendWorks\MendWorks\backend"
pip install -r requirements.txt
python app.py
```

Optional (use venv instead of global):
```
cd "C:\Cursor Projects\MendWorks\MendWorks\backend"
py -3.11 -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
python app.py
```

Default ports:
- Frontend: http://localhost:5173
- Backend: http://localhost:5000

### Supabase Database
Run migrations using the Supabase SQL Editor by pasting each file (order matters). Key files:
- `20250912055225_lingering_shape.sql` – creates `devices` (no CT scanner type)
- `20250913204500_create_eeg_predictions_table.sql` – `eeg_predictions`
- `20250114000001_create_mri_predictions_table.sql` – `mri_predictions`
- `20250114000002_create_ventilator_predictions_table.sql` – `ventilator_predictions`

If MRI saves fail, verify the tables exist. A helper check script `check_mri_table.sql` is included.

### Backend API (selected)
- `GET /api/health` – health check
- `POST /api/predict-eeg`
- `POST /api/predict-mri`
- `POST /api/predict-ventilator`
- `POST /api/delete-device` – deletes device and related prediction rows

The backend currently does not validate Supabase JWTs by default; Supabase is used directly by the frontend for auth and RLS. Add JWT verification if you expose the API publicly.

### Notes on Models
Model files (`.pth`, `.pkl`) are committed for reproducible inference and are below the 100MB GitHub limit. The backend loads them from `allmodels/` and `eegmodel/`.

### Common Issues
- "Invalid time value" on the dashboard: caused by invalid/empty timestamps; fixed with guards
- "Rendered fewer hooks than expected": ensure no early returns before all hooks
- 404 on API calls: confirm the Flask server is running on port 5000
- Delete device returns "Device not found": ensure the device exists for the current hospital under RLS

### Contributing & Scripts
- Linting is configured via the project’s ESLint setup
- Tailwind and PostCSS already configured
- Please avoid committing `backend/venv/` and OS/IDE files (see `.gitignore`)

### License
MIT


