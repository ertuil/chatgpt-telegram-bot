import asyncio
import base64
import os
import logging
import shelve
import time
import traceback
from typing import Any, Dict, Union
import aiohttp
from telegram import File, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import openai
from telegram.ext._application import BT
from telegram.error import RetryAfter, NetworkError, BadRequest
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    ArxivQueryRun,
    WikipediaQueryRun,
    WolframAlphaQueryRun,
)
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    WolframAlphaAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
    TextRequestsWrapper,
)

from dotenv import load_dotenv

load_dotenv()
sqlite_engine = create_async_engine(
    "sqlite+aiosqlite:///data/sqlite.db", echo=True, future=True
)

ADMIN_ID = os.environ.get("TELEGRAM_ADMIN_ID")
ADMIN_ID = int(ADMIN_ID)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-3.5-turbo")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)


TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
PAGE_LIMIT = 1500
TOTAL_WEB_LIMIT = 8000
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
FIRST_BATCH_DELAY = 1
VISION_MODEL = "gpt-4-vision-preview"

telegram_last_timestamp = None
telegram_rate_limit_lock = asyncio.Lock()


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


def is_chinese(string):
    for ch in string:
        if "\u4e00" <= ch <= "\u9fff":
            return "cn"
    return False


async def get_model(
    model: str = DEFAULT_MODEL, language: str = "en"
) -> RunnableWithMessageHistory:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a helpful ChatGPT Telegram bot. Answer as concisely as possible. Current Beijing Time: f{current_time}",
            ),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm = ChatOpenAI(
        model=model,
        base_url=OPENAI_BASE_URL,
        streaming=True,
        temperature=0.7,
        cache=True,
    )
    tools = [
        ArxivQueryRun(),
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(doc_content_chars_max=8000)),
    ]

    if (
        os.environ.get("GOOGLE_CSE_ID", None) is None
        or os.environ.get("GOOGLE_API_KEY", None) is None
    ):
        tools.append(
            DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=10))
        )
    else:
        search_wrapper = GoogleSearchAPIWrapper()

        google_search = Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=search_wrapper.run,
        )
        tools.append(google_search)

    if os.environ.get("WOLFRAM_ALPHA_APPID", None) is not None:
        tools.append(
            WolframAlphaQueryRun(
                api_wrapper=WolframAlphaAPIWrapper(
                    wolfram_alpha_appid=os.environ.get("WOLFRAM_ALPHA_APPID")
                )
            )
        )

    class SmallSizeRequestsWrapper(TextRequestsWrapper):
        async def _aget_resp_content(
            self, response: aiohttp.ClientResponse
        ) -> Union[str, Dict[str, Any]]:
            if self.response_content_type == "text":
                content = await response.text()
                if len(content) > 16000:
                    return content[:16000]
                return content
            elif self.response_content_type == "json":
                return await response.json()
            else:
                raise ValueError(f"Invalid return type: {self.response_content_type}")

    toolkit = RequestsToolkit(
        requests_wrapper=SmallSizeRequestsWrapper(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
            }
        ),
        allow_dangerous_requests=True,
    )
    tools.append(toolkit.get_tools()[0])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    chain = agent_executor  # | StrOutputParser()
    # chain_with_history = RunnableWithMessageHistory(
    #     chain,
    #     lambda session_id: SQLChatMessageHistory(
    #         session_id=session_id, connection=sqlite_engine, async_mode=True
    #     ),
    #     input_messages_key="question",
    #     history_messages_key="history",
    # ) | StrOutputParser()
    return chain


async def get_vision_model(model: str = VISION_MODEL):
    llm = (
        ChatOpenAI(
            model=model, base_url=OPENAI_BASE_URL, streaming=True, temperature=0.7
        )
        | StrOutputParser()
    )
    return llm


@only_whitelist
async def reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sender_id = update.message.from_user.id
    msg_id = update.message.message_id
    text = update.message.text
    reply_to_message = update.message.reply_to_message
    reply_to_id = None
    session_id = None
    model = DEFAULT_MODEL

    try:
        p = update.message.photo[-1]
        logging.info(f"[photo] get photo {p.file_size} {p.width}x{p.height}")
        obj: File = await p.get_file()
        image = await obj.download_as_bytearray()
        image = bytes(image)
        cap = update.message.caption
        text = cap if cap is not None and text is None else text
    except Exception:
        image = None

    if (
        reply_to_message is not None
        and update.message.reply_to_message.from_user.id == bot_id
    ):  # user reply to bot message
        reply_to_id = reply_to_message.message_id
        await pending_reply_manager.wait_for((chat_id, reply_to_id))
        key = repr(("session", chat_id, reply_to_id))
        if key not in db:
            logging.error(
                "Session message not found (chat_id=%r, msg_id=%r)",
                chat_id,
                reply_to_id,
            )
            return
        session_id = db[key]
    elif text.startswith("!") or text.startswith("！"):  # new message
        text = text[1:]
        session_id = f"{chat_id}_{msg_id}"
        db[repr(("session", chat_id, msg_id))] = session_id
    else:  # not reply or new message to bot
        if (
            update.effective_chat.id != update.message.from_user.id
        ):  # if not in private chat, do not send hint
            return
        session_id = f"{chat_id}_{msg_id}"

    logging.info(
        "New message: chat_id=%r, sender_id=%r, msg_id=%r, text=%r, session_id=%r",
        chat_id,
        sender_id,
        msg_id,
        text,
        session_id,
    )
    chain_with_history = await get_model(model=model)
    if image is not None:
        image_b64 = base64.b64encode(image).decode("ascii")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        prompt = [
            (
                "system",
                f"You are a helpful ChatGPT Telegram bot. Answer as concisely as possible. Current Beijing Time: {current_time}",
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ]
            ),
        ]
        vision_model = await get_vision_model(model=VISION_MODEL)
        async with BotReplyMessages(chat_id, msg_id, f"[{model}] ") as replymsgs:
            try:
                reply = await vision_model.ainvoke(prompt)
                history = SQLChatMessageHistory(
                    session_id=session_id, connection=sqlite_engine, async_mode=True
                )
                logging.info(f"History: {await history.aget_messages()}")
                await history.aadd_message(HumanMessage(content=text))
                await history.aadd_message(AIMessage(content=reply))
                await replymsgs.update(reply)
                await replymsgs.finalize()
                key = repr(("session", chat_id, replymsgs.replied_msgs[0][0]))
                db[key] = session_id
                logging.debug(f"insert session chat_id{key}: {session_id}")
            except Exception as e:
                logging.exception(
                    "Error (chat_id=%r, msg_id=%r): %s",
                    chat_id,
                    msg_id,
                    e,
                )
        return

    error_cnt = 0
    while True:
        reply = ""
        async with BotReplyMessages(chat_id, msg_id, f"[{model}] ") as replymsgs:
            try:
                history = SQLChatMessageHistory(
                    session_id=session_id, connection=sqlite_engine, async_mode=True
                )
                logging.info(f"History: {await history.aget_messages()}")
                stream = chain_with_history.astream(
                    {"question": text, "history": await history.aget_messages()},
                )
                first_update_timestamp = None
                action_logs = []
                async for delta in stream:
                    logging.debug(f"debug delta: {delta}")
                    for k, v in delta.items():
                        if k == "output":
                            reply += v
                            if first_update_timestamp is None:
                                first_update_timestamp = time.time()
                            if (
                                time.time()
                                >= first_update_timestamp + FIRST_BATCH_DELAY
                            ):
                                await replymsgs.update(reply + " [!Generating...]")
                        if k == "steps":
                            for ass in v:
                                action_logs.append(
                                    ass.action.log.replace("\n", "").replace("\r", "")
                                )
                await history.aadd_message(HumanMessage(content=text))
                await history.aadd_message(AIMessage(content=reply))
                if len(action_logs) > 0:
                    reply += "\n【日志】"
                    reply += "\n".join(action_logs)
                await replymsgs.update(reply)
                await replymsgs.finalize()
                key = repr(("session", chat_id, replymsgs.replied_msgs[0][0]))
                db[key] = session_id
                logging.debug(f"insert session chat_id{key}: {session_id}")
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
                will_retry = will_retry = (
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

    sqlLogger = logging.getLogger("sqlalchemy.engine.Engine")
    sqlLogger.setLevel(logging.WARNING)
    httpLogger = logging.getLogger("HTTP")
    httpLogger.setLevel(logging.WARNING)

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
            MessageHandler(
                (filters.TEXT | filters.PHOTO) & (~filters.COMMAND), reply_handler
            )
        )
        application.add_handler(CommandHandler("ping", ping))
        application.add_handler(CommandHandler("add_whitelist", add_whitelist_handler))
        application.add_handler(CommandHandler("del_whitelist", del_whitelist_handler))
        application.add_handler(CommandHandler("get_whitelist", get_whitelist_handler))
        application.run_polling()
