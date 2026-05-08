# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements first (to optimize Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]