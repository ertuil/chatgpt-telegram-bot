FROM python:3.13-alpine
RUN apk add libstdc++
WORKDIR /app
RUN pip install uv
ADD . /app
RUN addgroup gpt && adduser -h /app -D -G gpt -u 1000 -s /bin/false gpt
RUN mkdir /app/workspace
RUN chown -R gpt:gpt /app
USER gpt
RUN uv sync && uv tool install basic-memory
CMD ["uv", "run", "main.py"]

