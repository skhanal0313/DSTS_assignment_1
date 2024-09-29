# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /usr/src/part_b

# Copy the current directory contents into the container at /app
#COPY . /app

# Install the required Python packages
RUN pip install --no-cache-dir pandas scikit-learn joblib

COPY . .

EXPOSE 8000

# Run the Python script when the container launches
CMD ["python", "part_b.py"]