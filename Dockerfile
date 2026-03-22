FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Expose both UI and API ports
EXPOSE 8501 8000

# Default: run Streamlit UI
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]
