FROM python:3.10
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
# Adjust to match your actual startup command and port (Hugging Face default is 7860)
CMD ["uvicorn", "app.backend.api:app", "--host", "0.0.0.0", "--port", "7860"]