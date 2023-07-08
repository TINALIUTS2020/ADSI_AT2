# Use the official Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python code file to the container
COPY main.py .

# Expose the port that the FastAPI app will listen on
EXPOSE 8000

# Start the FastAPI app when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
