FROM python:3.9-alpine

WORKDIR /app

RUN pip install --no-cache-dir flask

COPY app.py .

CMD ["python", "app.py", "--port", "8080"]
