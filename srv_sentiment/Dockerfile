FROM python:3.10

COPY requirements.txt .
COPY app.py .

CMD ["pip", "install", "-r", "requirements.txt"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
# behind a TLS Proxy 
# CMD ["uvicorn", "app.app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
