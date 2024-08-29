# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install any dependencies specified in pyproject.toml
RUN poetry install --no-root --no-dev

# Expose the port that the Flask app will run on
EXPOSE 5000

# Run app.py when the container launches
CMD ["poetry", "run", "python", "app.py"]