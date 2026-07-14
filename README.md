# SkyForecast

SkyForecast is an airline passenger forecasting project built around a trained
LSTM model. This repo now includes two ways to demo it:

- `app.py` keeps the original Streamlit interface.
- `frontend/` and `backend/` provide the flagship React + GSAP portfolio site
  backed by a FastAPI model API.

## Local Setup

Use the existing Python virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Install the React dependencies:

```bash
cd frontend
npm install
```

Create a frontend environment file:

```bash
cp frontend/.env.example frontend/.env
```

Required key:

```bash
VITE_API_URL=http://localhost:8000
```

## Run The FastAPI Backend

From the project root:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000
```

Endpoints:

- `GET /api/history` returns the historical passenger series.
- `GET /api/metrics` returns MAE, MSE, and RMSE from the existing evaluator.
- `POST /api/forecast` accepts `{ "months": 12 }` and returns forecast values
  plus an illustrative RMSE-based uncertainty band.

## Run The React Frontend

In another terminal:

```bash
cd frontend
npm run dev
```

Open:

```bash
http://localhost:5173
```

## Run The Streamlit App

From the project root:

```bash
.venv/bin/python -m streamlit run app.py
```
