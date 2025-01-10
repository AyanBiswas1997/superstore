FROM python:3.8-slim-buster

# Set working directory
WORKDIR /service

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install  -r requirements.txt

# Copy the rest of the application files
COPY . .

# Command to run the application
CMD ["python3", "app.py"]
