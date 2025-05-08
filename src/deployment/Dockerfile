# Use lightweight Python base image
FROM python:3.9-slim


# Create app directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run the API
CMD ["python", "app.py"]
