FROM python:3.13-alpine
WORKDIR /app
ADD requirements.txt /app
RUN pip install -r requirements.txt
ADD . /app
