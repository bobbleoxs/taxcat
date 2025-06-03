# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# PORT will be provided by Railway at runtime
# ENV PORT 8000 

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# (e.g., for some Haystack backends or other libraries)
# Add any other system dependencies your project might need
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Install pipenv if you were using it (based on previous interactions, using requirements.txt)
# RUN pip install pipenv

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes your FastAPI app, scripts, and the data directory
COPY . .

# # Add the startup script
# COPY start.sh /start.sh
# RUN chmod +x /start.sh

# # Ensure the cache directory for joblib exists and is writable if needed by your app
# # RUN mkdir -p cache && chmod -R 777 cache

# # The command to run your application (FastAPI with Uvicorn)
# # Railway will inject the PORT environment variable.
# # Uvicorn will bind to 0.0.0.0 to be accessible from outside the container.
# # Use shell form for CMD to allow environment variable substitution for $PORT
# CMD ["/start.sh"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
