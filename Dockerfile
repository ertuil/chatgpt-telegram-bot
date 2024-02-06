FROM python:3.11-alpine
WORKDIR /app
ADD requirements.txt /app
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
ADD . /app
