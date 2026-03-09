FROM python:3.11-slim

WORKDIR /code

# Copy requirements first for better caching
COPY requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application code
COPY api.py /code/api.py
COPY src /code/src
COPY models /code/models

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
