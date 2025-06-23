FROM python:3.13-alpine
WORKDIR /app
ADD requirements.txt /app
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/simple -r requirements.txt
ADD . /app
