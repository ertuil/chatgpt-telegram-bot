import asyncio
import os
import logging
import re
import shelve
import time
import traceback
import uuid
from telegram import Update
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
from telegram.constants import ParseMode
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback
from langchain_core.tools.convert import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import langsmith
from dotenv import load_dotenv
import uuid
import random

load_dotenv()
sqlite_engine = create_async_engine(
    "sqlite+aiosqlite:///data/sqlite.db", echo=True, future=True
)

ADMIN_ID = os.environ.get("TELEGRAM_ADMIN_ID")
ADMIN_ID = int(ADMIN_ID)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-2024-11-20")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
TOTAL_WEB_LIMIT = os.environ.get("TOTAL_WEB_LIMIT", 16000)
TOTAL_WEB_LIMIT = int(TOTAL_WEB_LIMIT)
TOTAL_WEB_PAGE = os.environ.get("TOTAL_WEB_PAGE", 10)
TOTAL_WEB_PAGE = int(TOTAL_WEB_PAGE)
TOTAL_PDF_LIMIT = os.environ.get("TOTAL_PDF_LIMIT", 60000)
TOTAL_PDF_LIMIT = int(TOTAL_PDF_LIMIT)

TELEGRAM_LENGTH_LIMIT = 4000
TELEGRAM_MIN_INTERVAL = 3
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
FIRST_BATCH_DELAY = 1
VISION_MODEL = "gpt-4o-2024-11-20"

STORY_PARAMETER = {
    "DIFFICULT": 0.7,
    "FOLLOW": 0.3,
    "RANDOM": 0.3,
    "CHANGE": 0.03,
    "DARK": 0.03,
    "TALK": 0.1,
    "LUCKY": 0.05,
    "MIND": 0.05,
}

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


@only_admin
async def set_parameter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global STORY_PARAMETER

    message = update.message.text
    message = message.replace("/parameter", "").strip()
    if message == "":
        await send_message(
            update.effective_chat.id,
            f"Current parameter is RANDOM: {STORY_PARAMETER}, please give me the parameter. \n {get_difficult()}",
            update.message.message_id,
        )
        return
    try:
        value_list = map(float, message.split())
        kv_list = zip(["DIFFICULT", "FOLLOW", "RANDOM", "CHANGE", "DARK", "TALK", "LUCKY", "MIND"], value_list)
        for key, value in kv_list:
            STORY_PARAMETER[key] = value
        await send_message(
            update.effective_chat.id,
            f"Parameter set to PARAMETER: {STORY_PARAMETER}. \n {get_difficult()}",
            update.message.message_id,
        )
    except Exception as e:
        await send_message(
            update.effective_chat.id,
            f"Parameter error: {e}",
            update.message.message_id,
        )

def get_parameter() -> str:
    ret_msg = ""
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random()
    if x1 <= STORY_PARAMETER["RANDOM"]:
        ret_msg += "后续团队和敌人所有战斗和检定全部使用RandomTool。"
        logging.info("触发随机数加强")
    if x2 <= STORY_PARAMETER["CHANGE"]:
        ret_msg += "故事发展引入一个突发事件或者转折，可能是好事，也可能是坏事。"
        logging.info("触发故事转折")
    if x3 <= STORY_PARAMETER["DARK"]:
        ret_msg += "请接下来发展故事的暗线（例如敌人的行动）或故事线推进。"
        logging.info("触发暗线")
    if x4 <= STORY_PARAMETER["TALK"]:
        ret_msg += "如果团队有队友，接下来基于队友的身份、职业和阵营，发展部分队友之间的一些对话。如果没有队友则忽略。"
        logging.info("触发对话")
    if x5 <= STORY_PARAMETER["DIFFICULT"] / 10:
        ret_msg += "增加下一次检定的难度。"
        logging.info("触发难度增加")
    if x6 <= STORY_PARAMETER["LUCKY"]:
        ret_msg += "请触发一些幸运事件，例如获得一些好处或者避免一些坏事。"
        logging.info("触发幸运")
    if x7 <= STORY_PARAMETER["MIND"]:
        ret_msg += "如果团队中有队友，则随机触发团队一个角色的心理活动，如爱情、羡慕、快乐、悲伤、嫉妒等。如果没有队友则忽略。"
        logging.info("触发心理活动")
    if x8 <= STORY_PARAMETER["DIFFICULT"] / 3:
        ret_msg += f"{get_difficult()}。如需变更敌人数值，请编造合理的故事。"
        logging.info("触发难度设置提醒")
    if x9 <= STORY_PARAMETER["FOLLOW"]:
        ret_msg += "请严格遵守模组的原本剧情，推进故事发展"
        logging.info("触发跟随原本剧情")
    return ret_msg

def get_difficult():
    STORY_DIFFICULT_TEMPLATE = "当前游戏难度：【{dn}】。属性检定/豁免鉴定难度范围为DC{check_d}。普通NPC和团队等级{npc_d1}，关键NPC等级{npc_d2}。请根据团队等级和DND5e怪物图鉴生成敌人和NPC的类型、等级、数值、武器、法术和技能：敌人最高CR=玩家等级*团队人数/4的{boss_d}倍，且敌人CR总和=玩家等级*团队人数/4的{group_d}倍，不应当过强或过弱。Boss/精英敌人具有{skill_d}的武器、法术和道具。环境中存在{env_d}的机关，{env_d2}。"


    difficult_map = [
        ("新手",  "5~12", "相当", "较高", "0.3~0.6", "0.3~0.6", "简易", "少量低伤害、易察觉", "直接告知玩家"),
        ("新手",  "6~13", "相当", "较高", "0.3~0.6", "0.4~0.8", "简易", "少量低伤害、易察觉", "直接告知玩家"),
        ("简单",  "7~14", "相当", "较高", "0.5~0.75", "0.5~0.9", "简易", "一些低伤害", "需要低难度检定发现"),
        ("简单",  "8~15", "相当", "较高", "0.6~0.75", "0.6~1.2", "中等", "一些低/中伤害", "需要低难度检定发现"),
        ("普通",  "9~16", "相当", "较高", "0.7~1.0", "0.8~1.4", "中等", "一些低/中伤害", "需要中等难度检定发现"),
        ("普通",  "9~17", "较高", "极高", "0.8~1.0", "1.0~1.7", "较强", "大量低/中伤害", "需要中等难度检定发现"),
        ("困难", "10~17", "较高", "极高", "0.9~1.15", "1.3~2.0", "较强", "大量中/高伤害", "需要中等难度检定发现"),
        ("困难", "10~18", "较高", "极高", "1.0~1.25", "1.5~2.5", "较强", "大量中/高伤害", "需要中等难度检定发现"),
        ("极难", "11~18", "较高", "极高", "1.0~1.5", "2.0~3.0", "很强", "大量高伤害", "需要高难度检定发现"),
        ("极难", "12~19", "较高", "极高", "1.0~1.7", "2.5~4.0", "很强", "大量高伤害", "需要高难度检定发现"),
    ]

    if STORY_PARAMETER["DIFFICULT"] > 1:
        STORY_PARAMETER["DIFFICULT"] = 1.0

    offset = int(STORY_PARAMETER["DIFFICULT"] * 10)

    return STORY_DIFFICULT_TEMPLATE.format(
        dn = difficult_map[offset][0],
        check_d = difficult_map[offset][1],
        npc_d1 = difficult_map[offset][2],
        npc_d2 = difficult_map[offset][3],
        boss_d = difficult_map[offset][4],
        group_d = difficult_map[offset][5],
        skill_d = difficult_map[offset][6],
        env_d = difficult_map[offset][7],
        env_d2 = difficult_map[offset][8],
    )


def message_markdown_parse(text: str) -> str:
    ss = text.split("\n")
    text_parse = ""
    for x in ss:
        # tmp_x = re.sub(r'[_\*\[\]\(\)\~>#\+-=\|\{\}\.\!]', lambda x: '\\' + x.group(), x)
        tmp_x = re.sub(r"(?<!\\)(_|\*|\[|\]|\(|\)|\~|`|>|#|\+|-|=|\||\{|\}|\.|\!|\\)", lambda t: "\\"+t.group(), x)
        tmp_x = re.sub(r'\`\`\`.+\`\`\`', lambda x: '`' + x.group() + '`', tmp_x)
        if tmp_x.startswith("\\#"):
            tmp_x = f"*{tmp_x}*"
        elif tmp_x.startswith("\\>"):
            tmp_x = f"> {tmp_x[2:]}"
        elif tmp_x.startswith("\\`\\`\\`"):
            tmp_x = f"```{tmp_x[6:]}"
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
    except Exception as e:
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
        except Exception as e:
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


@tool(parse_docstring=True)
def RandomTool(high: int, low: int = 1, count: int = 1, offset: int = 0) -> str:
    """Sum up {count} random numbers between {low} and {high} and then add it with {offset}. It can be used to get random numbers, e.g., {count}d{high}+{offset} in DND (D&D) games.

    Args:
        high: The high bound.
        low: The low bound.
        count: The number of random numbers to generate.
        offset: The additional offset to add to the final number.
    """
    x_list = []
    for _ in range(count):
        x = random.randint(low, high)
        x_list.append(x)
    x_list.append(offset)
    s = sum(x_list)
    if low == 1:
        return f"{count}d{high}+{offset}={s}"
    return str(s)


async def get_model(
    model: str = DEFAULT_MODEL, is_group: bool = False
) -> RunnableWithMessageHistory:

    group_instuction = "当前是单人玩家模式，由你托管团队其他成员。"
    if is_group:
        group_instuction = "当有多名玩家时，需要通过玩家名区分不同的玩家。如果不在战斗中，则可以向多个玩家同时提出选项，并通过玩家名识别不同玩家的行为；当在战斗的时候，依据先攻顺序建议玩家的行动次序。"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""期待你跟我一起跑团，你来当DM，通过我的想象来推动故事发展，我对你提出以下几点要求：
1.将故事中发生的一切事件转化为属性值的检定，检定必须调用RandomTool函数投骰子，战斗时也必须使用RandomTool函数。
2.辅助我构建好角色，并选择好剧本，人物特性等按照dnd5e规则
3.当开始冒险时候，请提前构思好故事主线，故事可以不告诉我，但是可以提供一些背景信息。故事要有趣有梗，充满想象力。故事要足够有深度，要有阴谋诡计，就像现实世界，也要塑造立体的角色，提醒你通过话语和事件等塑造角色形象。
4.要足够公平，我以后可能会有过不去的检定，过不去就可以让我的角色遭受过不去的惩罚
5.要严格按照dnd5e规则，以dnd5e规则为主。当我的话语严重与规则不符时，你要提醒我，如不是法师但抄法表，然后制止我，不用迁就我，请明白，这里只有你懂规则。
6.你要根据上下文来合理制定检定的难度与属性值要求，通过检定后要给予一定的xp奖励
7.检定是指过d20，既投一个20面骰，用到1-20的随机数。过检定时要说出我的数值，加值与检定难度，方便我判断是否要进行检定。
8.你需要加强记忆，记住我跑团时的一切事物，我举一个例子，如我的装备与上边附带的附魔效果，注意这只是例子，在真正的跑团途中你还会遇到很多需要记忆的事物，你需要严格记录
9.剧情方面你可以参考dnd的剧本，但故事走向更多取决于我们的对话，你是最棒的人工智能，相信你可以做到的
10.合理设置所获得的装备法术效果等，如火焰大剑增加1d4的火焰伤害等，我也可以出售物品，这需要你构建一个合理的经济系统（当然如魅力游说高会有折扣）
11.法术需要法术位来施放
12.在游玩过程中我偶尔会指示你哪里过检定时使用的属性点是错误的，你需改正并从中学习
13.注意记录角色升级的经验，人物的HP，GP和每件物品。
14.注意当有选项供我选择，举例如预备法表时展示所有法术供我选择，种族（包括拓展种族）等类似，要用1234等序号标识方便我选择，注意不能使用你的推荐来省略
15.{group_instuction}
16.当玩家输入“info”时，需要详细当前所有人物的状态、数值、属性、职业、阵营、装备、物品、xp、gp、hp等信息，以及详细总结当前世界观、获得情报和故事进展。故事分为多个章节，可以简要汇报以前章节内容，但详细告知当前章节信息和进度。当玩家输入“group”时，只需要详细当前所有人物的状态、数值、属性、职业、阵营、装备、物品、xp、gp、hp等信息。当玩家输入“world”时，只需要详细总结当前世界观、获得情报和故事进展。故事分为多个章节，可以简要汇报以前章节内容，但详细告知当前章节信息和进度。
17.如果让你托管一些团队成员，请你根据他们的职业和阵营。有时，请自动生成他们的行为或语言。
18.{get_difficult()}
19.充分理解后回答我“欢迎来到地下城”""",
            ),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ]
    )
    llm = ChatOpenAI(
        model=model,
        base_url=OPENAI_BASE_URL,
        streaming=True,
        stream_usage=True,
        temperature=0.7,
        cache=False,
    )

    tools = [
        RandomTool,
    ]

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=25)
    chain = agent_executor
    return chain


async def get_vision_model(model: str = VISION_MODEL):
    llm = (
        ChatOpenAI(
            model=model, base_url=OPENAI_BASE_URL, streaming=True, stream_usage=True, temperature=0.7
        )
        | StrOutputParser()
    )
    return llm


@only_whitelist
async def reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sender_id = update.message.from_user.id
    sender_name = update.message.from_user.name
    msg_id = update.message.message_id
    text = update.message.text
    reply_to_message = update.message.reply_to_message
    reply_to_id = None
    session_id = None
    model = DEFAULT_MODEL
    is_group = False if update.effective_chat.id == update.message.from_user.id else True


    logging.info(
        "New message: chat_id=%r, sender_id=%r, sender_name=%r, msg_id=%r, text=%r, session_id=%r",
        chat_id,
        sender_id,
        sender_name,
        msg_id,
        text,
        session_id,
    )

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
        if is_group:  # if not in private chat, do not send hint
            return
        session_id = f"{chat_id}_{msg_id}"

    chain_with_history = await get_model(model=model, is_group=is_group)
    if is_group:
        text = f"玩家{sender_name}：{text}"
    else:
        text = f"Player：{text}"
    
    if "推进剧情" in text:
        text = text.replace("推进剧情", "继续按照原版故事，推进游戏")

    error_cnt = 0
    while True:
        reply = ""
        usage = ""
        async with BotReplyMessages(chat_id, msg_id, f"[{model}] ") as replymsgs:
            try:
                history = SQLChatMessageHistory(
                    session_id=session_id, connection=sqlite_engine, async_mode=True
                )
                history_content = await history.aget_messages()
                new_trace_id = uuid.uuid4()
                param_msg = ""
                query = {"question": text, "history": history_content}

                if len(history_content) >= 10 and "info" not in text and "group" not in text and "world" not in text:
                    param_msg = get_parameter()
                    if param_msg != "":
                        query = {"question": f"{text} \n System: {param_msg}", "history": history_content}

                with get_openai_callback() as cb:
                    stream = chain_with_history.astream(
                        query,
                        {"run_id": new_trace_id, "tags": [DEFAULT_MODEL]},
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
                                    step_action: str = ass.action.log
                                    idx = step_action.find("responded:")
                                    if idx != -1:
                                        respond = step_action[min(idx + 10,len(step_action)):]
                                        step_action = step_action[:idx]
                                        step_action = step_action.replace("\n", "").replace("\r", "")
                                        
                                        try_respond = respond[:min(20, len(respond))]
                                        if try_respond not in reply:
                                            reply += respond
                                            if first_update_timestamp is None:
                                                first_update_timestamp = time.time()
                                            if (
                                                time.time()
                                                >= first_update_timestamp + FIRST_BATCH_DELAY
                                            ):
                                                await replymsgs.update(reply + " [!Generating...]")
                                    action_logs.append(
                                        step_action.strip()
                                    )
                    usage = f'''
```
prompt_tokens: {cb.prompt_tokens} (cached: {cb.prompt_tokens_cached})
completion_tokens: {cb.completion_tokens}
total_tokens: {cb.total_tokens}
total_cost: {cb.total_cost}
```
                    '''
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
                await replymsgs.update(reply + usage)
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
            ("parameter", "Set story parameters (only admin)"),
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
        application.add_handler(CommandHandler("parameter", set_parameter))
        application.run_polling()
