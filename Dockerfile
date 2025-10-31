# ---- 1. Base image ----
FROM python:3.11-slim

# ---- 2. Install dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 3. Copy the rest of the app ----
COPY . .

# ---- 4. Streamlit configuration ----
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

# ---- 5. Run Streamlit ----
CMD ["streamlit", "run", "app.py"]
