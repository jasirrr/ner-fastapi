# First Stage: Build environment
FROM python:3.11-slim AS builder
WORKDIR /app

# Increase pip timeout and use a faster mirror
ENV PIP_DEFAULT_TIMEOUT=1000
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Install large dependencies first
RUN pip install --no-cache-dir torch==2.6.0 spacy==3.8.4

# Second Stage: Final image
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 10000

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
