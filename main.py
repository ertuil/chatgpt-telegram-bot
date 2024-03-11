import asyncio
import os
import logging
import shelve
import datetime
import time
import json
from urllib.parse import urlparse
import traceback
from typing import Dict, List, Optional, Tuple
import openai
from openai import AsyncOpenAI, OpenAI
import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import httpx
from telegram.error import RetryAfter, NetworkError, BadRequest
import re
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

import anthropic

ADMIN_ID = os.environ.get("TELEGRAM_ADMIN_ID")
ADMIN_ID = int(ADMIN_ID)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-3.5-turbo")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

TimeoutSetting = httpx.Timeout(15.0, read=15.0, write=15.0, connect=5.0)

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=TimeoutSetting)
sclient = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=TimeoutSetting)

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
PAGE_LIMIT = 1500
TOTAL_WEB_LIMIT = 8000
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
FIRST_BATCH_DELAY = 1

telegram_last_timestamp = None
telegram_rate_limit_lock = asyncio.Lock()

if os.environ.get("ANTHROPIC_API_KEY") is not None:
    logging.info("enabling anthropic claude 3")
    aapi_key = os.environ.get("ANTHROPIC_API_KEY")
    aclient = anthropic.AsyncAnthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=aapi_key,
        max_retries=OPENAI_MAX_RETRY,
        timeout=15,
    )
else:
    logging.info("anthropic claude 3 is not enabled")
    aclient = None


def google_search(search_term, **kwargs) -> List[Dict[str, str]]:
    if GOOGLE_API_KEY is None or GOOGLE_CSE_ID is None:
        logging.error("google search api key or cse id is not set")
        return []
    logging.info(f"google search: {search_term}")
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    try:
        res = service.cse().list(q=search_term, cx=GOOGLE_CSE_ID, **kwargs).execute()
        result = []
        items = res["items"]
        for item in items[: min(len(items), TOTAL_WEB_LIMIT // PAGE_LIMIT + 2)]:
            logging.info(
                f"google result: {item['title']} url: {item['link']} snippet: {item.get('snippet', '')}"
            )
            elem = {
                "title": item["title"],
                "snippet": item.get("snippet", ""),
                "url": item["link"],
            }
            result.append(elem)
        return result
    except Exception as e:
        traceback.print_exc()
        logging.error(f"goole search error: {e}")
        return []


def is_valid_url(url):

    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url_list = re.findall(regex, url)
    url_list = [x[0] for x in url_list]

    ret_list = []
    for url in url_list:
        try:
            parsed_uri = urlparse(url)
        except ValueError:
            continue
        ret_list.append(url)
    return ret_list


async def async_crawler(search_dict: Dict[str, str]) -> Optional[List[str]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67",
        "referer": "https://www.google.com/",
    }
    async with httpx.AsyncClient(
        headers=headers, timeout=TimeoutSetting, follow_redirects=True
    ) as client:
        try:
            url = search_dict["url"]
            title = search_dict["title"]
            snippet = search_dict["snippet"]

            response = await client.get(url)
            # 处理响应数据
            context_para = [f"Title: {title}", f"Snippet: {snippet}", "Content:"]
            logging.info(f"crawler get {title} {url}")

            soup = BeautifulSoup(response.text, "html.parser")
            for elem in soup.find_all(
                ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "article"]
            ):
                text = elem.get_text(strip=True, separator=" ")
                if len(text) <= 10:
                    continue
                context_para.append(text)

            article_msg = " ".join(context_para)
            return [url, title, snippet, article_msg]
        except Exception as e:
            logging.error(f"get {url} error: {str(e)}")
            return None


async def async_url(url: str) -> Tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67",
        "referer": "https://www.google.com/",
    }
    async with httpx.AsyncClient(
        headers=headers, timeout=TimeoutSetting, follow_redirects=True
    ) as client:
        try:

            response = await client.get(url)
            # 处理响应数据
            try:
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.text

                msg = f"Title: {title}\tContent: "
                for elem in soup.find_all(
                    ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "article"]
                ):
                    text = elem.get_text(strip=True, separator=" ")
                    if len(text) <= 6:
                        continue
                    msg += text
                    if len(msg) > TOTAL_WEB_LIMIT * 0.9:
                        break
            except Exception:
                title = ""
            return title, msg
        except Exception as e:
            logging.error(f"get {url} error: {str(e)}")
            return None


def is_chinese(string):
    for ch in string:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def get_query_question(query_list: List[str]):
    if len(query_list) == 1 and len(query_list[0]) < 15:
        return query_list[0]

    current_question = query_list[-1]
    history_question = query_list[:-1] if len(query_list) > 1 else []

    model = DEFAULT_MODEL
    query_msg = f"current question: {current_question}"
    if len(history_question) > 0:
        history_msg = "history questions: " + "\t".join(history_question)
        query_msg = history_msg + "\n" + query_msg
    prompt = f"""Conclude a concise Google Search question of the following within 15 words. Always search by keywords and don't repeat user's question in search query. Search for \"Bill Gates\" instead of \"Who is Bill Gates\" when user asks \"Who is Bill Gates?\".

{query_msg}

"""
    if is_chinese(query_msg):
        prompt += "请用简体中文回复"
    message = [{"role": "system", "content": prompt}]
    logging.info(f"Summary request: {prompt}")

    resp = sclient.chat.completions.create(
        model=model,
        messages=message,
    )
    try:
        query = resp.choices[0].message.content
        query = query.strip().replace("\n", "").replace('"', "")
    except Exception as e:
        logging.error(f"get_query_question error: {e}")
    logging.info(f"Summary question: {query}")
    return query


def PROMPT(model, context_set: List[List[str]] = [], language="English"):

    mode_str = "ChatGPT"
    mode_company = "OpenAI"
    if "claude-3" in model:
        mode_str = "Claude 3"
        mode_company = "Anthropic"

    if len(context_set) == 0:
        s = "You are {mode_str} Telegram bot. {mode_str} is a large language model trained by {mode_company}. Answer as concisely as possible. Knowledge cutoff: Sep 2021. Current Beijing Time: {current_time}"

    else:
        mode_str = "GPT-4"
        mode_company = "OpenAI"
        if "claude-3" in model:
            mode_str = "Claude 3"
            mode_company = "Anthropic"
        s = """Web search results:

{web_results}
Current Beijing date: {current_time}

Instructions: You are {mode_str} Telegram bot, trained by {mode_company}. Answer as concisely as possible. You can use the provided web search results if you do not know the answer. Make sure to cite results using [^index] (e.g. [^1] [^2]) notation after the reference.
Reply in {reply_language} language.
"""
        websearch_list = []
        for idx, elem in enumerate(context_set):
            url = elem[0]
            content = elem[3]
            item = f"[{idx+1}] url: {url}  {content}"
            websearch_list.append(item)
        web_results = "\n".join(websearch_list)
        s = s.replace("{web_results}", web_results)

    s = s.replace(
        "{current_time}",
        (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    s = s.replace("{reply_language}", language)
    s = s.replace("{mode_str}", mode_str)
    s = s.replace("{mode_company}", mode_company)
    return s


class PendingReplyManager:
    def __init__(self):
        self.messages = {}

    def add(self, reply_id):
        assert reply_id not in self.messages
        self.messages[reply_id] = asyncio.Event()

    def remove(self, reply_id):
        if reply_id not in self.messages:
            return
        self.messages[reply_id].set()
        del self.messages[reply_id]

    async def wait_for(self, reply_id):
        if reply_id not in self.messages:
            return
        logging.info("PendingReplyManager waiting for %r", reply_id)
        await self.messages[reply_id].wait()
        logging.info("PendingReplyManager waiting for %r finished", reply_id)


def within_interval():
    global telegram_last_timestamp
    if telegram_last_timestamp is None:
        return False
    remaining_time = telegram_last_timestamp + TELEGRAM_MIN_INTERVAL - time.time()
    return remaining_time > 0


def ensure_interval(interval=TELEGRAM_MIN_INTERVAL):
    def decorator(func):
        async def new_func(*args, **kwargs):
            async with telegram_rate_limit_lock:
                global telegram_last_timestamp
                if telegram_last_timestamp is not None:
                    remaining_time = telegram_last_timestamp + interval - time.time()
                    if remaining_time > 0:
                        await asyncio.sleep(remaining_time)
                result = await func(*args, **kwargs)
                telegram_last_timestamp = time.time()
                return result

        return new_func

    return decorator


def retry(max_retry=30, interval=10):
    def decorator(func):
        async def new_func(*args, **kwargs):
            for _ in range(max_retry - 1):
                try:
                    return await func(*args, **kwargs)
                except (RetryAfter, NetworkError) as e:
                    if isinstance(e, BadRequest):
                        raise
                    logging.exception(e)
                    await asyncio.sleep(interval)
            return await func(*args, **kwargs)

        return new_func

    return decorator


def is_whitelist(chat_id):
    whitelist = db["whitelist"]
    return chat_id in whitelist


def add_whitelist(chat_id):
    whitelist = db["whitelist"]
    whitelist.add(chat_id)
    db["whitelist"] = whitelist


def del_whitelist(chat_id):
    whitelist = db["whitelist"]
    whitelist.discard(chat_id)
    db["whitelist"] = whitelist


def get_whitelist():
    return db["whitelist"]


def only_admin(func):
    async def new_func(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return
        if update.message.from_user.id != ADMIN_ID:
            await send_message(
                update.effective_chat.id,
                "Only admin can use this command",
                update.message.message_id,
            )
            return
        await func(update, context)

    return new_func


def only_private(func):
    async def new_func(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return
        if update.effective_chat.id != update.message.from_user.id:
            await send_message(
                update.effective_chat.id,
                "This command only works in private chat",
                update.message.message_id,
            )
            return
        await func(update, context)

    return new_func


def only_whitelist(func):
    async def new_func(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return
        if not is_whitelist(update.effective_chat.id):
            if update.effective_chat.id == update.message.from_user.id:
                await send_message(
                    update.effective_chat.id,
                    "This chat is not in whitelist",
                    update.message.message_id,
                )
            return
        await func(update, context)

    return new_func


async def completion(
    chat_history, model, chat_id, msg_id, search: bool = False
):  # chat_history = [user, ai, user, ai, ..., user]
    assert len(chat_history) % 2 == 1

    new_chat_history = []
    for chat in chat_history:
        if "【开始回答】" in chat:
            chat = re.sub(r".*【开始回答】", "", chat, flags=re.DOTALL)
        if "【参考资料】" in chat:
            chat = re.sub(r"【参考资料】.*", "", chat, flags=re.DOTALL)
        new_chat_history.append(chat)
    chat_history = new_chat_history
    logging.info("chat_history: %r", chat_history)
    last_question = chat_history[-1]

    content_set: List[List[str]] = []
    tmp_content_set: List[List[str]] = []

    result: List[Dict[str, str]] = []
    if GOOGLE_API_KEY is not None and GOOGLE_CSE_ID is not None and search:
        context_len = 0
        try:
            maybe_url_list = is_valid_url(last_question)
            print(maybe_url_list)
            if len(maybe_url_list) > 0:
                maybe_url = maybe_url_list[0]
                is_url = True
            else:
                maybe_url = last_question
                is_url = False
        except Exception:
            is_url = False

        if is_url:
            ## get url
            logging.info("get url: %s", maybe_url)
            try:
                title, content = await async_url(maybe_url)
                content = content[: min(len(content), int(TOTAL_WEB_LIMIT * 0.9))]
                content_set.append([maybe_url, title, "", content])
            except Exception as e:
                logging.error(f"get url error {e}")
        else:
            ## start Google Search
            # step 1: search google
            try:
                question_list = [
                    c for idx, c in enumerate(chat_history) if idx % 2 == 0
                ]
                google_question = question_list[-1]
                yield f"【谷歌搜索】 {google_question}\n"
                result = google_search(google_question)
            except Exception as e:
                logging.error(f"get google error {e}")

            # step 2: get webpage content
            tasks = [async_crawler(search_dict) for search_dict in result]
            crawler_results = await asyncio.gather(*tasks)
            tmp_content_set = [elem for elem in crawler_results if elem is not None]

            # step 3: length limit
            for elem in tmp_content_set:
                url, title, snippet, content = elem[0], elem[1], elem[2], elem[3]
                try:
                    if len(content) > PAGE_LIMIT:
                        content = content[:PAGE_LIMIT] + "..."

                    if context_len + len(content) < TOTAL_WEB_LIMIT:
                        content_set.append([url, title, snippet, content])
                        context_len += len(content)
                    else:
                        break
                except Exception:
                    traceback.print_exc()

    logging.info(f"context set: {content_set}")
    language = "简体中文" if is_chinese(last_question) else "English"
    prompt = PROMPT(model, content_set, language)
    if is_chinese(last_question):
        prompt += "请用简体中文回复"

    if "claude-3" in model and aclient is not None:
        messages = []
    else:
        messages = [{"role": "system", "content": prompt}]
    ll = len(prompt)

    roles = ["user", "assistant"]
    # role_id = 0

    temp_msg_pairs = []
    idx = 0
    while idx < len(chat_history):
        temp_msg_pair = []
        if idx == len(chat_history) - 1:
            temp_msg_pair.append({"role": roles[0], "content": chat_history[idx]})
        else:
            temp_msg_pair.append({"role": roles[0], "content": chat_history[idx]})
            temp_msg_pair.append({"role": roles[1], "content": chat_history[idx + 1]})
        idx += 2
        temp_msg_pairs.append(temp_msg_pair)

    temp_msg_pairs.reverse()
    temp_msgs = []
    for temp_msg_pair in temp_msg_pairs:
        c = 0
        for temp_msg in temp_msg_pair:
            c += len(temp_msg["content"])
        ll += c
        if ll > TOTAL_WEB_LIMIT and len(temp_msgs) > 0:
            break
        if len(temp_msg_pair) == 1:
            temp_msgs.append(temp_msg_pair[0])
        else:
            temp_msgs.append(temp_msg_pair[1])
            temp_msgs.append(temp_msg_pair[0])
    temp_msgs.reverse()
    messages.extend(temp_msgs)
    # for msg in chat_history:
    #     messages.append({"role": roles[role_id], "content": msg})
    #     role_id = 1 - role_id

    logging.info(
        "Request (model=%s,chat_id=%r, msg_id=%r): %s", model, chat_id, msg_id, messages
    )

    if "claude-3" in model and aclient is not None:
        if "$$$" in last_question:
            q_s = last_question.split("$$$")
            if len(q_s) >= 2:
                prompt = q_s[0]
                query = q_s[1]
                messages[-1]["content"] = query
                logging.info(f"claude 3 prompt: {prompt} {messages}")
        stream = await aclient.messages.create(
            model=model, messages=messages, stream=True, max_tokens=4096, system=prompt
        )
        full_answer = ""
        async for event in stream:
            logging.info("Response (chat_id=%r, msg_id=%r): %s", chat_id, msg_id, event)
            if event.type == "message_start":
                assert event.message.role == "assistant"
                assert event.message.content == []
                assert event.message.stop_reason is None
            elif event.type == "content_block_delta":
                assert event.index == 0
                assert event.delta.type == "text_delta"
                full_answer += event.delta.text
                yield event.delta.text
            elif event.type == "content_block_start":
                assert event.index == 0
                assert event.content_block.text == ""
            elif event.type == "content_block_stop":
                assert event.index == 0
            elif event.type == "message_delta":
                stop_reason = event.delta.stop_reason
                if stop_reason is not None:
                    if stop_reason == "end_turn":
                        full_answer = full_answer.replace("^]", "]")
                        ref_list = re.findall(r"\[\^\d\]", full_answer)
                        if len(ref_list) > 0:
                            yield "\n\n【参考资料】\n"
                            dref_list = []
                            for idx, ref in enumerate(ref_list):
                                try:
                                    dref = (
                                        ref.replace("]", "")
                                        .replace("[", "")
                                        .replace("^", "")
                                        .strip()
                                    )
                                    dref = int(dref) - 1
                                    dref_list.append((ref, dref))
                                except Exception:
                                    traceback.print_exc()
                            dref_list = list(set(dref_list))
                            dref_list.sort(key=lambda x: x[0])
                            for ref, dref in dref_list:
                                yield f"{ref} {content_set[dref][1]} {content_set[dref][0]}\n"
                    else:
                        yield f'\n\n[!] Error: stop_reason="{stop_reason}"'
    else:
        stream = await client.chat.completions.create(
            model=model, messages=messages, stream=True
        )

        yield "【开始回答】"

        full_answer = ""
        async for response in stream:
            # logging.info(
            #     "Response (chat_id=%r, msg_id=%r): %s",
            #     chat_id,
            #     msg_id,
            #     response,
            # )

            obj = response.choices[0]
            if obj.finish_reason is not None:
                if obj.finish_reason == "length":
                    yield " [!Output truncated due to limit]"
                full_answer = full_answer.replace("^]", "]")
                ref_list = re.findall(r"\[\^\d\]", full_answer)
                if len(ref_list) > 0:
                    yield "\n\n【参考资料】\n"
                    dref_list = []
                    for idx, ref in enumerate(ref_list):
                        try:
                            dref = (
                                ref.replace("]", "")
                                .replace("[", "")
                                .replace("^", "")
                                .strip()
                            )
                            dref = int(dref) - 1
                            dref_list.append((ref, dref))
                        except Exception:
                            traceback.print_exc()
                    dref_list = list(set(dref_list))
                    dref_list.sort(key=lambda x: x[0])
                    for ref, dref in dref_list:
                        yield f"{ref} {content_set[dref][1]} {content_set[dref][0]}\n"
                return
            if "role" in obj.delta:
                if obj.delta.role != "assistant":
                    raise ValueError("Role error")
            if obj.delta.content is not None:
                full_answer += obj.delta.content
                yield obj.delta.content


def construct_chat_history(chat_id, msg_id):
    messages = []
    should_be_bot = False
    model = DEFAULT_MODEL
    while True:
        key = repr((chat_id, msg_id))
        if key not in db:
            logging.error(
                "History message not found (chat_id=%r, msg_id=%r)", chat_id, msg_id
            )
            return None, None
        is_bot, text, reply_id, *params = db[key]
        if params:
            model = params[0]
        if is_bot != should_be_bot:
            logging.error(
                "Role does not match (chat_id=%r, msg_id=%r)", chat_id, msg_id
            )
            return None, None
        messages.append(text)
        should_be_bot = not should_be_bot
        if reply_id is None:
            break
        msg_id = reply_id
    if len(messages) % 2 != 1:
        logging.error(
            "First message not from user (chat_id=%r, msg_id=%r)", chat_id, msg_id
        )
        return None, None
    return messages[::-1], model


async def get_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_whitelist(update.effective_chat.id):
        await send_message(
            update.effective_chat.id, "Not in the whitelist", update.message.message_id
        )
        return

    mode_msg = """Available modes:
- ! or ！: Use GPT-3.5 Turbo
- !! or ！！: Use GPT-4 Turbo
- g! or g！: Use GPT-4 Turbo with Google search
"""

    if aclient is not None:
        mode_msg += "- c! or c！: Use Claude 3 Opus\n"
        mode_msg += "- cs! or cs！: Use Claude 3 Sonnet\n"
        mode_msg += "- cg! or cg！: Use Claude 3 Opus with Google search\n"
    await send_message(update.effective_chat.id, mode_msg, update.message.message_id)


@only_admin
async def add_whitelist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_whitelist(update.effective_chat.id):
        await send_message(
            update.effective_chat.id, "Already in whitelist", update.message.message_id
        )
        return
    add_whitelist(update.effective_chat.id)
    await send_message(
        update.effective_chat.id, "Whitelist added", update.message.message_id
    )


@only_admin
async def del_whitelist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_whitelist(update.effective_chat.id):
        await send_message(
            update.effective_chat.id, "Not in whitelist", update.message.message_id
        )
        return
    del_whitelist(update.effective_chat.id)
    await send_message(
        update.effective_chat.id, "Whitelist deleted", update.message.message_id
    )


@only_admin
@only_private
async def get_whitelist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_message(
        update.effective_chat.id, str(get_whitelist()), update.message.message_id
    )


@retry()
@ensure_interval()
async def send_message(chat_id, text, reply_to_message_id):
    logging.info(
        "Sending message: chat_id=%r, reply_to_message_id=%r, text=%r",
        chat_id,
        reply_to_message_id,
        text,
    )
    msg = await application.bot.send_message(
        chat_id,
        text,
        reply_to_message_id=reply_to_message_id,
        disable_web_page_preview=True,
    )
    logging.info(
        "Message sent: chat_id=%r, reply_to_message_id=%r, message_id=%r",
        chat_id,
        reply_to_message_id,
        msg.message_id,
    )
    return msg.message_id


@retry()
@ensure_interval()
async def edit_message(chat_id, text, message_id):
    logging.info(
        "Editing message: chat_id=%r, message_id=%r, text=%r", chat_id, message_id, text
    )
    try:
        await application.bot.edit_message_text(
            text,
            chat_id=chat_id,
            message_id=message_id,
            disable_web_page_preview=True,
        )
    except BadRequest as e:
        if (
            e.message
            == "Message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message"
        ):
            logging.info(
                "Message not modified: chat_id=%r, message_id=%r", chat_id, message_id
            )
        else:
            raise
    else:
        logging.info("Message edited: chat_id=%r, message_id=%r", chat_id, message_id)


@retry()
@ensure_interval()
async def delete_message(chat_id, message_id):
    logging.info("Deleting message: chat_id=%r, message_id=%r", chat_id, message_id)
    try:
        await application.bot.delete_message(
            chat_id,
            message_id,
        )
    except BadRequest as e:
        if e.message == "Message to delete not found":
            logging.info(
                "Message to delete not found: chat_id=%r, message_id=%r",
                chat_id,
                message_id,
            )
        else:
            raise
    else:
        logging.info("Message deleted: chat_id=%r, message_id=%r", chat_id, message_id)


class BotReplyMessages:
    def __init__(self, chat_id, orig_msg_id, prefix):
        self.prefix = prefix
        self.msg_len = TELEGRAM_LENGTH_LIMIT - len(prefix)
        assert self.msg_len > 0
        self.chat_id = chat_id
        self.orig_msg_id = orig_msg_id
        self.replied_msgs = []
        self.text = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, type, value, tb):
        await self.finalize()
        for msg_id, _ in self.replied_msgs:
            pending_reply_manager.remove((self.chat_id, msg_id))

    async def _force_update(self, text):
        slices = []
        while len(text) > self.msg_len:
            slices.append(text[: self.msg_len])
            text = text[self.msg_len :]
        if text:
            slices.append(text)
        if not slices:
            slices = [""]  # deal with empty message

        for i in range(min(len(slices), len(self.replied_msgs))):
            msg_id, msg_text = self.replied_msgs[i]
            if slices[i] != msg_text:
                await edit_message(self.chat_id, self.prefix + slices[i], msg_id)
                self.replied_msgs[i] = (msg_id, slices[i])
        if len(slices) > len(self.replied_msgs):
            for i in range(len(self.replied_msgs), len(slices)):
                if i == 0:
                    reply_to = self.orig_msg_id
                else:
                    reply_to, _ = self.replied_msgs[i - 1]
                msg_id = await send_message(
                    self.chat_id, self.prefix + slices[i], reply_to
                )
                self.replied_msgs.append((msg_id, slices[i]))
                pending_reply_manager.add((self.chat_id, msg_id))
        if len(self.replied_msgs) > len(slices):
            for i in range(len(slices), len(self.replied_msgs)):
                msg_id, _ = self.replied_msgs[i]
                await delete_message(self.chat_id, msg_id)
                pending_reply_manager.remove((self.chat_id, msg_id))
            self.replied_msgs = self.replied_msgs[: len(slices)]

    async def update(self, text):
        self.text = text
        if not within_interval():
            await self._force_update(self.text)

    async def finalize(self):
        await self._force_update(self.text)


@only_whitelist
async def reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sender_id = update.message.from_user.id
    msg_id = update.message.message_id
    text = update.message.text
    logging.info(
        "New message: chat_id=%r, sender_id=%r, msg_id=%r, text=%r",
        chat_id,
        sender_id,
        msg_id,
        text,
    )
    reply_to_message = update.message.reply_to_message
    reply_to_id = None
    model = DEFAULT_MODEL
    search = False
    if (
        reply_to_message is not None
        and update.message.reply_to_message.from_user.id == bot_id
    ):  # user reply to bot message
        reply_to_id = reply_to_message.message_id
        await pending_reply_manager.wait_for((chat_id, reply_to_id))
    elif (
        text.startswith("!")
        or text.startswith("！")
        or text.startswith("g")
        or text.startswith("c")
    ):  # new message
        if text.startswith("!!") or text.startswith("！！"):
            text = text[2:]
            model = "gpt-4-turbo-preview"
        elif text.startswith("g!") or text.startswith("g！"):
            text = text[2:]
            model = "gpt-4-turbo-preview"
            search = True
        elif text.startswith("c!") or text.startswith("c！"):
            text = text[2:]
            model = "claude-3-opus-20240229"
        elif text.startswith("cs!") or text.startswith("cs！"):
            text = text[3:]
            model = "claude-3-sonnet-20240229"
        elif text.startswith("cg!") or text.startswith("cg！"):
            text = text[3:]
            model = "claude-3-opus-20240229"
            search = True
        else:
            text = text[1:]
    else:  # not reply or new message to bot
        if (
            update.effective_chat.id != update.message.from_user.id
        ):  # if not in private chat, do not send hint
            return
        # await send_message(
        #     update.effective_chat.id,
        #     "Please start a new conversation with ! or reply to a bot message",
        #     update.message.message_id,
        # )
    db[repr((chat_id, msg_id))] = (False, text, reply_to_id, model)
    chat_history, model = construct_chat_history(chat_id, msg_id)
    if chat_history is None:
        await send_message(
            update.effective_chat.id,
            f"[!] Error: Unable to proceed with this conversation. Potential causes: the message replied to may be incomplete, contain an error, be a system message, or not exist in the database.",
            update.message.message_id,
        )
        return

    error_cnt = 0
    while True:
        reply = ""
        async with BotReplyMessages(chat_id, msg_id, f"[{model}] ") as replymsgs:
            try:
                stream = completion(chat_history, model, chat_id, msg_id, search)
                first_update_timestamp = None
                async for delta in stream:
                    reply += delta
                    if first_update_timestamp is None:
                        first_update_timestamp = time.time()
                    if time.time() >= first_update_timestamp + FIRST_BATCH_DELAY:
                        await replymsgs.update(reply + " [!Generating...]")
                await replymsgs.update(reply)
                await replymsgs.finalize()
                for message_id, _ in replymsgs.replied_msgs:
                    db[repr((chat_id, message_id))] = (True, reply, msg_id, model)
                return
            except Exception as e:
                error_cnt += 1
                logging.exception(
                    "Error (chat_id=%r, msg_id=%r, cnt=%r): %s",
                    chat_id,
                    msg_id,
                    error_cnt,
                    e,
                )
                will_retry = (
                    not isinstance(e, openai.APIStatusError)
                    and error_cnt <= OPENAI_MAX_RETRY
                )
                error_msg = (
                    f"[!] Error: {traceback.format_exception_only(e)[-1].strip()}"
                )
                if will_retry:
                    error_msg += f"\nRetrying ({error_cnt}/{OPENAI_MAX_RETRY})..."
                if reply:
                    error_msg = reply + "\n\n" + error_msg
                await replymsgs.update(error_msg)
                if will_retry:
                    await asyncio.sleep(OPENAI_RETRY_INTERVAL)
                if not will_retry:
                    break


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_message(
        update.effective_chat.id,
        f"chat_id={update.effective_chat.id} user_id={update.message.from_user.id} is_whitelisted={is_whitelist(update.effective_chat.id)}",
        update.message.message_id,
    )


async def post_init(application):
    await application.bot.set_my_commands(
        [
            ("ping", "Test bot connectivity"),
            ("add_whitelist", "Add this group to whitelist (only admin)"),
            ("del_whitelist", "Delete this group from whitelist (only admin)"),
            ("get_whitelist", "List groups in whitelist (only admin)"),
        ]
    )


if __name__ == "__main__":
    logFormatter = logging.Formatter(
        "%(asctime)s %(process)d %(levelname)s %(message)s"
    )

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("chatgpt-telegram-bot.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    with shelve.open("data/db") as db:
        if "whitelist" not in db:
            db["whitelist"] = {ADMIN_ID}
        bot_id = int(TELEGRAM_BOT_TOKEN.split(":")[0])
        pending_reply_manager = PendingReplyManager()
        application = (
            ApplicationBuilder()
            .token(TELEGRAM_BOT_TOKEN)
            .post_init(post_init)
            .concurrent_updates(True)
            .build()
        )
        application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), reply_handler)
        )
        application.add_handler(CommandHandler("ping", ping))
        application.add_handler(CommandHandler("mode", get_mode))
        application.add_handler(CommandHandler("add_whitelist", add_whitelist_handler))
        application.add_handler(CommandHandler("del_whitelist", del_whitelist_handler))
        application.add_handler(CommandHandler("get_whitelist", get_whitelist_handler))
        application.run_polling()
