FROM python:3.13-alpine
WORKDIR /app
ADD requirements.txt /app
RUN pip install -i https://mirrors.bfsu.edu.cn/pypi/web/simple -r requirements.txt
ADD . /app
