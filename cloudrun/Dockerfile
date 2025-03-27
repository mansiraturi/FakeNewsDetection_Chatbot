# Use official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# This is critical: Cloud Run sets PORT env variable, we need to use it
ENV PORT=8080

# The key fix: make sure we're exposing the right port
EXPOSE ${PORT}

# Make sure we're using the PORT environment variable that Cloud Run provides
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0