# ---- 1) Base image ----
FROM python:3.11-slim

# Allow cache-busting from your trigger (optional)
ARG CACHE_BUST=initial

# ---- 2) OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ---- 3) Python deps ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prove no pathlib backport is present (shows in Cloud Build logs)
RUN python -c "import sys, pkgutil; print('pathlib backport present?' , any(m.name in ('pathlib','pathlib2') for m in pkgutil.iter_modules()))"

# ---- 4) Copy project ----
COPY . .

# Show which pathlib will be used (should print something like /usr/local/lib/python3.11/pathlib.py)
RUN python -c "import pathlib, sys; print('USING PATHLIB:', getattr(pathlib,'__file__', 'built-in'))"

# ---- 5) Build the serving pipeline (.pkl) at build time ----
# This will create saluSCORE_ped_pipeline.pkl inside the image
RUN python -c "from build_serving_pipeline import main; main()"

# ---- 6) Streamlit runtime ----
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080
CMD ["streamlit", "run", "app.py"]
