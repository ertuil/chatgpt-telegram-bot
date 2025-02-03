import asyncio
import base64
import json
import os
import logging
import re
import shelve
import time
import traceback
import uuid
from telegram import File, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode
import openai
from telegram.error import RetryAfter, NetworkError, BadRequest
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
import langsmith
from dotenv import load_dotenv

load_dotenv()
sqlite_engine = create_async_engine(
    "sqlite+aiosqlite:///data/sqlite.db", echo=True, future=True
)

ADMIN_ID = os.environ.get("TELEGRAM_ADMIN_ID")
ADMIN_ID = int(ADMIN_ID)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deepseek-chat")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
TOTAL_WEB_LIMIT = os.environ.get("TOTAL_WEB_LIMIT", 4000)
TOTAL_WEB_LIMIT = int(TOTAL_WEB_LIMIT)
TOTAL_WEB_PAGE = os.environ.get("TOTAL_WEB_PAGE", 5)
TOTAL_WEB_PAGE = int(TOTAL_WEB_PAGE)
TOTAL_PDF_LIMIT = os.environ.get("TOTAL_PDF_LIMIT", 60000)
TOTAL_PDF_LIMIT = int(TOTAL_PDF_LIMIT)

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
FIRST_BATCH_DELAY = 1
VISION_MODEL = "deepseek-chat"

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


@only_admin
async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global DEFAULT_MODEL
    message = update.message.text
    message = message.replace("/set_model", "").replace(" ", "").strip()
    if message == "":
        await send_message(
            update.effective_chat.id,
            f"Current model is {DEFAULT_MODEL}, please give me the model name",
            update.message.message_id,
        )
        return
    DEFAULT_MODEL = message
    await send_message(
        update.effective_chat.id, f"Model set to {message}", update.message.message_id
    )


def message_markdown_parse(text: str) -> str:
    ss = text.split("\n")
    text_parse = ""
    for x in ss:
        tmp_x = re.sub(r'[_*[\]()~>#\+\-=|{}.!]', lambda x: '\\' + x.group(), x)
        if tmp_x.startswith("\\#"):
            tmp_x = f"*{tmp_x}*"
        elif tmp_x.startswith("\\>"):
            tmp_x = f"> {tmp_x[2:]}"
        else: 
            tmp_xx = re.sub(r'\\\*\\\*.+\\\*\\\*', lambda x: '*' + x.group() + '*', tmp_x)
            if tmp_xx != tmp_x:
                tmp_x = tmp_xx
            else:
                tmp_x = re.sub(r'\\\*.+\\\*', lambda x: '_' + x.group() + '_', tmp_x)
        text_parse += tmp_x + "\n"
    return text_parse


@retry()
@ensure_interval()
async def send_message(chat_id, text, reply_to_message_id):
    logging.info(
        "Sending message: chat_id=%r, reply_to_message_id=%r, text=%r",
        chat_id,
        reply_to_message_id,
        text,
    )
    try:
        text_parse = message_markdown_parse(text)

        msg = await application.bot.send_message(
            chat_id,
            text_parse,
            reply_to_message_id=reply_to_message_id,
            disable_web_page_preview=True,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except ValueError or BadRequest as e:
        logging.warning(f"Fallback to text mode: {e}")
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
        try:
            text_parse = message_markdown_parse(text)

            await application.bot.edit_message_text(
                text_parse,
                chat_id=chat_id,
                message_id=message_id,
                disable_web_page_preview=True,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except ValueError or BadRequest as e:
            logging.warning(f"Fallback to text mode: {e}")
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


class GoogleSearchAPIWrapperSelf(GoogleSearchAPIWrapper):
    def run(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        metadata_results = []
        results = self._google_search_results(query, num=self.k)
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            metadata_result = {
                "title": result["title"],
                "link": result["link"],
            }
            if "snippet" in result:
                metadata_result["snippet"] = result["snippet"]
            metadata_results.append(metadata_result)
        return json.dumps(metadata_results, ensure_ascii=False)


async def get_model(
    model: str = DEFAULT_MODEL, language: str = "en"
) -> RunnableWithMessageHistory:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a helpful DeepSeek Telegram bot. Answer as concisely as possible. Current Beijing Time: f{current_time}",
            ),
            MessagesPlaceholder(variable_name="history", optional=True),
            MessagesPlaceholder(variable_name="pdf_content", optional=True),
            ("human", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ]
    )
    llm = ChatDeepSeek(
        model=model,
        base_url=OPENAI_BASE_URL,
        streaming=True,
        temperature=0.5,
        cache=False,
    )

    chain = prompt | llm  # | StrOutputParser()
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
        ChatDeepSeek(
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
    pdf_content = None

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

    try:
        f = update.message.document
        if f.mime_type == "application/pdf":
            logging.info(f"[pdf] get pdf {f.file_name} {f.file_size}")
            obj: File = await f.get_file()
            pdf = await obj.download_as_bytearray()
            pdf = bytes(pdf)
            cap = update.message.caption
            text = cap if cap is not None and text is None else text

            if not os.path.exists("data/upload"):
                os.makedirs("data/upload")
            path = f"data/upload/{uuid.uuid4()}-{f.file_name}"
            with open(path, "wb") as file:
                file.write(pdf)
            loader = PyPDFLoader(path)
            pdf_content = ""
            async for page in loader.alazy_load():
                pdf_content += page.page_content
            pdf_content = f"PDF File: {f.file_name}\nContent: {pdf_content[:min(len(pdf_content), TOTAL_PDF_LIMIT)]}"
            logging.info(f"[pdf] get pdf content {pdf_content}")
    except Exception as e:
        logging.exception(f"[pdf] get pdf error: {e}")
        pdf_content = None


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
    elif text.startswith("d!") or text.startswith("d！"):  # new message
        text = text[2:]
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
                f"You are a helpful DeepSeek Telegram bot. Answer as concisely as possible. Current Beijing Time: {current_time}",
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
        think = ""
        async with BotReplyMessages(chat_id, msg_id, f"[{model}] ") as replymsgs:
            try:
                history = SQLChatMessageHistory(
                    session_id=session_id, connection=sqlite_engine, async_mode=True
                )
                history_content = await history.aget_messages()
                logging.info(f"History: {history_content}")
                new_trace_id = uuid.uuid4()
                query = {"question": text, "history": history_content}
                if pdf_content is not None and pdf_content != "":
                    query["pdf_content"] = [SystemMessage(content=pdf_content)]
                stream = chain_with_history.astream(
                    query,
                    {"run_id": new_trace_id, "tags": [DEFAULT_MODEL]},
                )
                first_update_timestamp = None
                action_logs = []
                async for delta in stream:
                    logging.debug(f"debug delta: {delta}")
                    rv = delta.additional_kwargs.get("reasoning_content", "")
                    v = delta.content
                    if rv != "":
                        if think == "":
                            think = "【思考】"
                        think += rv
                    if v != "":
                        if not think.endswith("【回复】") and think != "":
                            think += "\n【回复】"
                        reply += v
                    if first_update_timestamp is None:
                        first_update_timestamp = time.time()
                    if (
                        time.time()
                        >= first_update_timestamp + FIRST_BATCH_DELAY
                    ):
                        await replymsgs.update(think + reply + " [!Generating...]")
                if pdf_messages := query.get("pdf_content", []):
                    if pdf_messages is not None and len(pdf_messages) >= 1:
                        pdf_m = pdf_messages[0]
                        await history.aadd_message(pdf_m)
                await history.aadd_message(HumanMessage(content=text))
                await history.aadd_message(AIMessage(content=reply))
                if len(action_logs) > 0:
                    reply += "\n【日志】"
                    reply += "\n".join(action_logs)
                    try:
                        if os.environ.get("LANGCHAIN_API_KEY", None) is not None:
                            client = langsmith.Client()
                            share_url = client.share_run(new_trace_id)
                            reply += f"\n日志地址：{share_url}"
                    except Exception as e:
                        logging.warning(f"get log error: {e}")
                await replymsgs.update(think + reply)
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
            ("set_model", "Set GPT model (only admin)"),
        ]
    )


if __name__ == "__main__":
    logFormatter = logging.Formatter(
        "%(asctime)s %(process)d %(levelname)s %(message)s"
    )

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

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
                (filters.TEXT | filters.PHOTO | filters.Document.PDF)
                & (~filters.COMMAND),
                reply_handler,
            )
        )
        application.add_handler(CommandHandler("ping", ping))
        application.add_handler(CommandHandler("add_whitelist", add_whitelist_handler))
        application.add_handler(CommandHandler("del_whitelist", del_whitelist_handler))
        application.add_handler(CommandHandler("get_whitelist", get_whitelist_handler))
        application.add_handler(CommandHandler("set_model", set_model))
        application.run_polling()
