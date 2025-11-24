FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libpq-dev gcc
RUN pip install pandas numpy sqlalchemy psycopg2-binary psutil
CMD ["tail", "-f", "/dev/null"]