FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libpq-dev gcc
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["tail", "-f", "/dev/null"]
