# Use the official Python 3.12 image from the Docker Hub
FROM python:3.12.4-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . ./

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
