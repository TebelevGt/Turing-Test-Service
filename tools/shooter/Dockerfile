# Use an official Python runtime as a base image
FROM python:3.8.20-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy data
COPY data /app/data

# Copy dependencies file (requirements.txt)
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source file
COPY main.py /app/main.py

# Run module
CMD ["python", "-m", "main"]
