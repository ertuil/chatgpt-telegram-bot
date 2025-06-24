FROM python:3.13-alpine
RUN apk add libstdc++
WORKDIR /app
RUN pip install uv
ADD . /app
RUN useradd -r -u 1000 -g gpt
RUN mkdir /app/workspace
RUN chown -R gpt:gpt /app
USER gpt
RUN uv sync
CMD ["uv", "run", "main.py"]

