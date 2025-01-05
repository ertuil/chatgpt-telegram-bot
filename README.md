# chatgpt-telegram-bot (with multiple plugins)

An Extended ChatGPT Telegram bot with the following plugins using LangChain and LangSmith.
* Arxiv
* Wikipedia
* DuckDuckGo
* GoogleSearch
* WolframAlpha

Also, it supports:
* Tracing steps using LangSmith
* GPT-4v for image understanding

## Download & Usage

Pull docker image:

```
docker pull ghcr.io/ertuil/chatgpt-telegram-bot:latest
```

Download `docker-compose.example.yml` as `docker-compose.yml`, and configure the environments.

```
docker-compose up -d
```



## Inspire

This bot is modified and is greatly based on [zzh1996's telegram bot](https://github.com/zzh1996/chatgpt-telegram-bot). Please refer to the original repository for information on its features and usage.