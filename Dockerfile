FROM python:3.13-alpine
WORKDIR /app
RUN pip install uv
ADD . /app
RUN uv sync
RUN mkdir /app/workspace