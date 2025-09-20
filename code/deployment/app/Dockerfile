# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app source code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck (optional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]