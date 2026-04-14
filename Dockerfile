# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install curl for health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY app.py .
COPY models/ ./models/
COPY Dataset/ ./Dataset/

# Expose default Hugging Face Space port
EXPOSE 7860

# Health check (supports Hugging Face dynamic PORT)
HEALTHCHECK CMD sh -c 'curl --fail "http://localhost:${PORT:-7860}/_stcore/health" || exit 1'

# Run Streamlit app (bind to Hugging Face runtime port)
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-7860} --server.address 0.0.0.0"]
