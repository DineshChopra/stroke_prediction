FROM python:3.8-slim-buster

COPY requirements.txt .

RUN pip install -r requirements.txt && \
	rm requirements.txt

EXPOSE 80

COPY /app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
