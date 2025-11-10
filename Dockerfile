# ---- 1. Base image ----
FROM python:3.11-slim

# Cache-buster you can change each deploy
ARG CACHE_BUST=initial

# ---- 2. OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ---- 3. Python deps ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: prove no backport pathlib is installed (shows in build logs)
RUN pip list | grep -E "^pathlib($|2)" || echo "OK: no pathlib backport installed"

# ---- 4. Copy the project ----
# (Make sure the training assets listed above are included in the repo)
COPY . .

# Sanity: show which pathlib will be used (should print .../lib/python3.11/pathlib.py)
RUN python - << 'PY'
import pathlib, inspect
print("USING PATHLIB:", inspect.getsourcefile(pathlib))
PY

# ---- 5. Build the serving pipeline (.pkl) at build time ----
# This calls your script which creates saluSCORE_ped_pipeline.pkl
RUN python -c "from build_serving_pipeline import main; main()"

# ---- 6. Streamlit runtime config ----
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8080

CMD ["streamlit", "run", "app.py"]
