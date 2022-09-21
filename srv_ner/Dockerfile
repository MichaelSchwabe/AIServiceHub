FROM python:3.10

COPY requirements.txt .
COPY app.py .

CMD ["pip", "install", "-r", "requirements.txt"]


# Large News and Media Model
CMD ["python", "-m", "spacy", "download", "de_core_news_lg"]

# Medium News and Media Model
# CMD ["python", "-m", "spacy", "download", "de_core_news_md"]
# Small News and Media Model
# CMD ["python", "-m", "spacy", "download", "de_core_news_sm"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
# behind a TLS Proxy 
# CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
