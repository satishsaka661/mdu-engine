FROM python:3.11-slim

WORKDIR /app

# Copy API requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the mdu_engine modules from project root
COPY mdu_engine/ ./mdu_engine/

# Copy the API application
COPY api/main.py .

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]