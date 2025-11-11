# ==========================================================
# 1. Base image
# ==========================================================
FROM python:3.11-slim

# Optional cache-buster to force rebuilds
ARG CACHE_BUST=initial

# ==========================================================
# 2. OS Dependencies
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# 3. Python Dependencies
# ==========================================================
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Check for accidental pathlib backport
RUN python -c "import pathlib, pkgutil; mods=[m.name for m in pkgutil.iter_modules()]; \
print('Pathlib backport present?', ('pathlib' in mods or 'pathlib2' in mods)); \
print('Using pathlib at:', getattr(pathlib, '__file__', 'builtin'))"

# ==========================================================
# 4. Copy full project
# ==========================================================
COPY . .

# ==========================================================
# 5. TRAIN ARTIFACTS INSIDE DOCKER
#    This script must output:
#      - normalization_scaler.joblib
#      - calibrated_xgboost_bagging_model.pkl
# ==========================================================
RUN python train_model_and_scaler.py

# ==========================================================
# 6. BUILD SERVING PIPELINE
#    This will create saluSCORE_ped_pipeline.pkl
# ==========================================================
RUN python -c "from build_serving_pipeline import main; main()"

# ==========================================================
# 7. Streamlit Runtime
# ==========================================================
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD ["streamlit", "run", "app.py"]
