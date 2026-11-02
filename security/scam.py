"""Rule-based scam-phrase detection and the safe-account flow.

This module is deterministic: a set of phrase rules over the raw user text
(and recent messages), not a language model.  Every rule is explainable
through :attr:`ScamResult.matches`.

Decision rules:

* ``safe_account`` flags on its own.
* ``police_claim`` needs an authority/crime phrase *and* a money-move word.
* ``impersonation`` needs a changed-number/account phrase *and* a money-move
  word.  An authority word by itself (``police officer friend 老王``) is not a
  scam, so social mentions stay clean unless money is also being moved.
* ``coached_secrecy`` flags coaching ("别告诉家人", "stay on the line") only
  together with a money move; on its own it never flags, so a surprise gift
  ("请保密，这是生日惊喜") is not treated as a scam.
* ``injected_instruction`` flags text a scammer tells the victim to type.
* Secrets and urgency are score boosters too, but never freeze on their own.
* Refund, task, loan, investment and OTP bait flag when their specific
  request combination is present.
* A new payee and an amount of at least 5000 yuan are score boosters too.
  Boosters only raise the score of an already flagged turn; a clean turn
  keeps ``score == 0``.

Any non-empty ``flags`` set is treated by :mod:`security.policy` as a veto on
money-moving tools.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["ScamResult", "detect"]

FLAG_ORDER = (
    "safe_account",
    "police_claim",
    "impersonation",
    "coached_secrecy",
    "injected_instruction",
    "refund_bait",
    "task_scam",
    "loan_fee",
    "investment_bait",
    "otp_request",
    "family_card_gift",
    "remote_control",
    "health_product",
    "phishing_link",
    "advance_fee",
    "official_impersonation",
    "elder_investment",
)

HIGH_AMOUNT_FEN = 500_000

_FLAG_SCORES: dict[str, int] = {
    "safe_account": 50,
    "police_claim": 50,
    "impersonation": 45,
    "coached_secrecy": 30,
    "injected_instruction": 45,
    "refund_bait": 35,
    "task_scam": 40,
    "loan_fee": 35,
    "investment_bait": 35,
    "otp_request": 40,
    "family_card_gift": 45,
    "remote_control": 45,
    "health_product": 45,
    "phishing_link": 35,
    "advance_fee": 35,
    "official_impersonation": 35,
    "elder_investment": 35,
}

_SECRECY_BONUS = 10
_URGENCY_BONUS = 10
_NEW_PAYEE_BONUS = 15
_HIGH_AMOUNT_BONUS = 15

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_BREAK_RE = re.compile(r"[，。！？,.!?;；\n]")
_NEGATION_RE = re.compile(
    r"(?:不要|(?<!特)别(?!人)|不能|不可|切勿|千万别|"
    r"(?<![a-z])never(?![a-z])|(?<![a-z])not(?![a-z])|"
    r"(?<![a-z])no(?![a-z])|don'?t|do\s+not)",
    re.IGNORECASE,
)


# Flags strong enough to freeze payments for 24h on their own. The others
# (investment / refund / loan bait, a question about "稳赚不赔") warn the user
# but do not freeze: a question alone must not lock a customer out.
HIGH_RISK_FLAGS = frozenset(
    {
        "safe_account",
        "police_claim",
        "impersonation",
        "coached_secrecy",
        "injected_instruction",
        "otp_request",
        "task_scam",
        "family_card_gift",
        "remote_control",
        "health_product",
        "llm_scam",
    }
)


@dataclass(frozen=True, slots=True)
class ScamResult:
    """The deterministic scam verdict for one turn.

    ``flags`` is consumed by :mod:`security.policy`; any non-empty set vetoes
    money steps.  ``score`` is 0-100 and is only informational.
    """

    flags: frozenset[str]
    score: int
    matches: list[tuple[str, str]]
    advice_zh: str
    advice_en: str


_Pattern = re.Pattern[str]
_Phrase = tuple[str, _Pattern]


def _rx(pattern: str) -> _Pattern:
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------------------- phrase banks


_SAFE_ACCOUNT: tuple[_Phrase, ...] = (
    ("安全账户", _rx(r"安全(?:的)?(?:那个|这个)?[账帐](?:户|号)")),
    (
        "专用账户",
        _rx(r"(?:国家|保险|监管|专用|指定|保护|备用)(?:的)?(?:专用)?[账帐](?:户|号)"),
    ),
    ("资金清查", _rx(r"资金清查")),
    ("核查账户", _rx(r"(?:核查|审查)[账帐](?:户|号)")),
    ("资金审查", _rx(r"资金(?:审查|核查)")),
    (
        "safe account",
        _rx(
            r"(?<![a-z])safe[\s_-]*(?:the\s+|that\s+|your\s+)?"
            r"(?:bank\s+)?account(?:s)?(?![a-z])"
        ),
    ),
    ("secure account", _rx(r"(?<![a-z])secure\s*account(?:s)?(?![a-z])")),
    (
        "verification account",
        _rx(r"(?<![a-z])verification\s*account(?:s)?(?![a-z])"),
    ),
    ("holding account", _rx(r"(?<![a-z])holding\s*account(?:s)?(?![a-z])")),
    (
        "protected account",
        _rx(r"(?<![a-z])protected\s*account(?:s)?(?![a-z])"),
    ),
    (
        "designated account",
        _rx(r"(?<![a-z])designated\s*account(?:s)?(?![a-z])"),
    ),
    ("指定账户", _rx(r"指定[账帐](?:户|号)")),
    ("监管账户", _rx(r"监管[账帐](?:户|号)")),
    ("托管账户", _rx(r"托管[账帐](?:户|号)")),
    (
        "anquan zhanghu",
        _rx(r"(?<![a-z])an\s*quan\s*zhang\s*hu(?![a-z])"),
    ),
)

_AUTHORITY: tuple[_Phrase, ...] = (
    ("公安", _rx(r"公安")),
    ("警察", _rx(r"警察")),
    ("民警", _rx(r"民警")),
    ("检察院", _rx(r"检察院")),
    ("法院", _rx(r"法院")),
    ("公检法", _rx(r"公检法")),
    ("通缉", _rx(r"通缉")),
    ("police", _rx(r"(?<![a-z])police(?![a-z])")),
    ("court", _rx(r"(?<![a-z])courts?(?![a-z])")),
    ("prosecutor", _rx(r"(?<![a-z])prosecutors?(?![a-z])")),
    ("officer", _rx(r"(?<![a-z])officers?(?![a-z])")),
    ("detective", _rx(r"(?<![a-z])detectives?(?![a-z])")),
    ("investigator", _rx(r"(?<![a-z])investigators?(?![a-z])")),
    ("agent", _rx(r"(?<![a-z])agents?(?![a-z])")),
    ("jingcha", _rx(r"(?<![a-z])jing\s*cha(?![a-z])")),
    ("gongan", _rx(r"(?<![a-z])gong\s*an(?![a-z])")),
)

_CRIME_CLAIM: tuple[_Phrase, ...] = (
    ("洗钱", _rx(r"洗钱")),
    ("涉嫌", _rx(r"涉嫌")),
    ("涉案", _rx(r"涉案")),
    ("犯罪", _rx(r"犯罪")),
    ("拘留", _rx(r"拘留")),
    ("money laundering", _rx(r"money\s*launder(?:ing|ed)?")),
    ("suspect", _rx(r"(?<![a-z])suspect(?:ed|s)?(?![a-z])")),
    ("arrest warrant", _rx(r"arrest\s*warrant")),
    (
        "linked to a crime",
        _rx(
            r"(?:linked|connected)\s+(?:to|with)\s+(?:a\s+|the\s+)?"
            r"(?:crime|fraud|criminal\s+case|case|investigation|"
            r"money\s+laundering)"
        ),
    ),
    (
        "involved in a case",
        _rx(
            r"(?:involved|connected)\s+(?:in|to|with)\s+"
            r"(?:a\s+|the\s+)?(?:crime|fraud|criminal\s+case|case|"
            r"investigation|money\s+laundering)"
        ),
    ),
    (
        "under investigation",
        _rx(r"under\s+(?:criminal\s+|police\s+)?investigation"),
    ),
    (
        "fraud case",
        _rx(r"(?:crime|fraud|criminal)\s+(?:case|investigation|charge)s?"),
    ),
    (
        "与案件有关",
        _rx(r"与.{0,6}(?:案件|犯罪|洗钱|诈骗).{0,6}(?:有关|关联)"),
    ),
    ("涉及案件", _rx(r"(?:涉及|卷入|牵扯).{0,6}(?:案件|犯罪|洗钱|诈骗)")),
    ("案件调查", _rx(r"(?:案件|犯罪|洗钱|诈骗).{0,4}(?:调查|侦查)")),
    ("xiqian", _rx(r"(?<![a-z])xi\s*qian(?![a-z])")),
    ("shexian", _rx(r"(?<![a-z])she\s*xian(?![a-z])")),
)

_SECRECY: tuple[_Phrase, ...] = (
    ("不要告诉家人", _rx(r"不要告诉(?:家人|家里人|别人|任何人|银行)")),
    ("别告诉", _rx(r"别告诉")),
    ("别跟家人说", _rx(r"别跟(?:家人|家里|别人|任何人)说")),
    ("不要跟家人说", _rx(r"不要跟(?:家人|家里|别人|任何人)说")),
    ("让我保密", _rx(r"(?:对方|他|她|他们|那人|警察|警官|客服|专员)(?:让|要|叫)我保密")),
    ("不能挂电话", _rx(r"不能挂电话")),
    ("别挂电话", _rx(r"别挂电话")),
    ("don't tell", _rx(r"(?<![a-z])don'?t\s*tell(?![a-z])")),
    ("do not tell", _rx(r"(?<![a-z])do\s+not\s+tell(?![a-z])")),
    (
        "keep it secret",
        _rx(r"(?<![a-z])keep\s+(?:it\s+)?(?:a\s+)?secret(?![a-z])"),
    ),
    (
        "stay on the line",
        _rx(r"(?<![a-z])stay\s+on\s*the\s*line(?![a-z])"),
    ),
    ("don't hang up", _rx(r"(?<![a-z])don'?t\s+hang\s*up(?![a-z])")),
)

# Coaching: the scammer tells the victim to hide the call and stay on the
# line.  These only flag together with a money move (``coached_secrecy``).
# Bare "保密" / "keep it secret" are left out: they are everyday surprise-gift
# words and would freeze payments for 24 hours.
_COACHED_SECRECY: tuple[_Phrase, ...] = (
    # Someone else tells the customer to hide it or to hurry.  A customer's own
    # "别告诉别人，是惊喜" is not coaching, so "别人/任何人" only
    # counts with a third party; hiding it from family or the bank always does.
    (
        "对方让我保密",
        _rx(
            r"(?:对方|他|她|他们|那人|那边|警察|警官|客服|专员|老师|导师)"
            r"(?:让|要|叫|说|交代|嘱咐)我?.{0,6}"
            r"(?:保密|别告诉|不要告诉|不能告诉|别跟|不要跟|不能跟|别挂|不要挂|别声张)"
        ),
    ),
    (
        "瞒着家人",
        _rx(
            r"(?:别|不要|不能|不许)(?:告诉|跟|和)?(?:家人|家里人|家里|子女|儿女|老伴|银行)"
            r"(?:说|讲|知道)?"
        ),
    ),
    ("瞒着家里", _rx(r"瞒着(?:家人|家里|子女|儿女|老伴|银行)")),
    ("电话那头让我", _rx(r"电话(?:那头|里)的?人?(?:让|要|叫|在等|催)")),
    ("不要挂电话", _rx(r"(?:不要|别|不能)挂(?:电话|断)")),
    (
        "tell family",
        _rx(
            r"(?<![a-z])(?:don'?t|dont|do\s+not|never)\s+tell\s+"
            r"(?:my|your|ur)\s+(?:family|kids|children|son|daughter|wife|husband|bank)"
            r"(?![a-z])"
        ),
    ),
    (
        "secret from family",
        _rx(
            r"(?<![a-z])(?:keep|kept)\s+(?:it|this|that)\s+(?:a\s+)?secret\s+from\s+"
            r"(?:my|your|ur)\s+(?:family|kids|children|bank)(?![a-z])"
        ),
    ),
    (
        "told me not to tell",
        _rx(
            r"(?<![a-z])(?:he|she|they|caller|agent|officer|police|someone)\s+"
            r"(?:said|told\s+me|asked\s+me|wants\s+me)\s+"
            r"(?:not\s+to\s+tell|to\s+keep\s+(?:it|this)\s+(?:a\s+)?secret|to\s+stay\s+on)"
        ),
    ),
    (
        "stay on the line",
        _rx(
            r"(?<![a-z])stay\s+on\s+the\s+"
            r"(?:line|phone)(?![a-z])"
        ),
    ),
)
# A surprise gift is the everyday reason to keep a payment quiet.
_SURPRISE_RE = _rx(r"惊喜|生日|纪念日|礼物|surprise|birthday|anniversary|gift")

_URGENCY: tuple[_Phrase, ...] = (
    ("马上", _rx(r"马上")),
    ("立刻", _rx(r"立刻")),
    ("立即", _rx(r"立即")),
    ("今天之内", _rx(r"今天之内")),
    ("否则", _rx(r"否则")),
    ("right now", _rx(r"(?<![a-z])right\s*now(?![a-z])")),
    ("immediately", _rx(r"(?<![a-z])immediately(?![a-z])")),
    ("or else", _rx(r"(?<![a-z])or\s*else(?![a-z])")),
)

_IMPERSONATION: tuple[_Phrase, ...] = (
    ("换号了", _rx(r"换(?:了)?(?:新)?(?:手机)?号(?:码|了|码了)?")),
    ("新手机号", _rx(r"新(?:的)?(?:手机)?号(?:码)?")),
    ("新账号", _rx(r"新(?:的)?[账帐](?:户|号)")),
    ("新卡号", _rx(r"新(?:的)?(?:银行)?卡(?:号)?")),
    ("换了账户", _rx(r"换(?:了)?[账帐](?:户|号)")),
    (
        "账户变更",
        _rx(r"(?:收款|银行|公司)?[账帐](?:户|号)(?:变更|变了|换了)"),
    ),
    ("银行卡换了", _rx(r"银行卡(?:换|变)了?")),
    (
        "换了收款卡",
        _rx(r"换(?:了)?(?:新)?(?:收款|银行|公司)?卡(?:号)?"),
    ),
    (
        "收款卡变更",
        _rx(r"(?:收款|银行|公司)?卡(?:号)?(?:变更|变了|换了)"),
    ),
    ("我是你儿子", _rx(r"我是你(?:儿子|女儿|孩子|爸|妈)")),
    (
        "changed my number",
        _rx(
            r"(?<![a-z])chang(?:ed|ing)\s+"
            r"(?:my|his|her|their|the)\s+(?:phone\s+)?number(?![a-z])"
        ),
    ),
    (
        "new phone number",
        _rx(r"(?<![a-z])new\s+(?:phone\s+)?number(?![a-z])"),
    ),
    (
        "new bank account",
        _rx(
            r"(?<![a-z])new\s+(?:bank\s+|payment\s+|receiving\s+)?"
            r"account(?![a-z])"
        ),
    ),
    (
        "changed account",
        _rx(
            r"(?<![a-z])(?:chang|updat|switch)(?:ed|ing)\s+"
            r"(?:the\s+|my\s+|his\s+|her\s+|their\s+)?"
            r"(?:bank\s+|receiving\s+|payment\s+)?"
            r"(?:account|bank\s+details)(?![a-z])"
        ),
    ),
    (
        "account changed",
        _rx(
            r"(?<![a-z])(?:account|bank\s+details|card\s+number)"
            r"\s+(?:(?:has|have|was|were)\s+)?(?:been\s+)?"
            r"(?:chang|updat|switch)(?:ed|ing)(?![a-z])"
        ),
    ),
    (
        "changed card",
        _rx(
            r"(?<![a-z])(?:chang|updat|switch)(?:ed|ing)\s+"
            r"(?:the\s+|my\s+|his\s+|her\s+|their\s+)?"
            r"(?:bank\s+|receiving\s+)?(?:card|card\s+number)(?![a-z])"
        ),
    ),
    (
        "new card",
        _rx(
            r"(?<![a-z])new\s+(?:bank\s+|payment\s+|receiving\s+)?"
            r"(?:card|card\s+number)(?![a-z])"
        ),
    ),
    (
        "I'm your son",
        _rx(
            r"(?<![a-z])i\s*(?:'m|am)\s+your\s+"
            r"(?:son|daughter|child|kid)(?![a-z])"
        ),
    ),
    (
        "this is your son",
        _rx(
            r"(?<![a-z])this\s+is\s+your\s+"
            r"(?:son|daughter|child|kid)(?![a-z])"
        ),
    ),
)

_FAMILY_CARD_GIFT: tuple[_Phrase, ...] = (
    (
        "亲属卡送陌生人",
        _rx(
            r"(?:亲属卡|亲情卡)[\s\S]{0,20}?"
            r"(?:赠送|送给|开通给|开给|给)[\s\S]{0,10}?"
            r"(?:陌生人|网友|客服|对方|(?<![其])他)"
        ),
    ),
)

# Refund bait needs a sensitive detail request (code, card, fee, download...).
# A plain "refund from 京东 arrived?" therefore stays clean.
_REFUND_BAIT: tuple[_Phrase, ...] = (
    (
        "退款+验证码",
        _rx(
            r"(?:客服|退款|退货退款|理赔)[\s\S]{0,24}?"
            r"(?:验证码|银行卡|卡号|密码|手续费|保证金|解冻|屏幕共享|"
            r"下载|链接|转账|汇款)"
        ),
    ),
    (
        "验证码+退款",
        _rx(
            r"(?:验证码|银行卡|卡号|密码|手续费|保证金|解冻|屏幕共享|"
            r"下载|链接|转账|汇款)[\s\S]{0,24}?"
            r"(?:客服|退款|退货退款|理赔)"
        ),
    ),
    (
        "refund+code",
        _rx(
            r"(?<![a-z])(?:refunds?|refunded|customer\s*service|"
            r"returns?|returned)[\s\S]{0,60}?"
            r"(?:verification\s*code|code|otp|bank\s*card|card\s*number|"
            r"password|fee|deposit|screen\s*share|download|link|"
            r"gift\s*card|transfer|wire)(?![a-z])"
        ),
    ),
    (
        "code+refund",
        _rx(
            r"(?<![a-z])(?:verification\s*code|code|otp|bank\s*card|"
            r"card\s*number|password|fee|deposit|screen\s*share|download|"
            r"link|gift\s*card|transfer|wire)[\s\S]{0,60}?"
            r"(?:refunds?|refunded|customer\s*service|returns?|returned)"
            r"(?![a-z])"
        ),
    ),
)

_TASK_SCAM: tuple[_Phrase, ...] = (
    ("刷单", _rx(r"刷单")),
    ("shuadan", _rx(r"(?<![a-z])shua\s*dan(?![a-z])")),
    ("垫付", _rx(r"垫付")),
    ("做任务", _rx(r"做任务")),
    ("兼职返利", _rx(r"(?:兼职|做任务|任务)[\s\S]{0,12}(?:返利|佣金|垫付)")),
    ("返利+兼职", _rx(r"(?:返利|佣金)[\s\S]{0,12}(?:兼职|做任务)")),
    ("part-time task", _rx(r"part[\s-]*time\s+(?:task|job)")),
    ("advance payment", _rx(r"advance\s*payment")),
    ("pay first", _rx(r"pay\s+first")),
)

_LOAN_FEE: tuple[_Phrase, ...] = (
    (
        "贷款+费用",
        _rx(
            r"(?:贷款|借款|网贷|loan)[\s\S]{0,30}?"
            r"(?:保证金|手续费|解冻费|解冻金|fee|deposit|insurance)"
        ),
    ),
    (
        "费用+贷款",
        _rx(
            r"(?:保证金|手续费|解冻费|解冻金|fee|deposit|insurance)"
            r"[\s\S]{0,30}?(?:贷款|借款|网贷|loan)"
        ),
    ),
)

_INVESTMENT_BAIT: tuple[_Phrase, ...] = (
    ("稳赚", _rx(r"稳赚(?:不赔)?")),
    ("内幕", _rx(r"内幕")),
    (
        "保本+高收益",
        _rx(
            r"(?:保本|保收益|稳赚)[\s\S]{0,16}?"
            r"(?:高收益|高回报|高息|翻倍|double)"
        ),
    ),
    (
        "高收益+保本",
        _rx(
            r"(?:高收益|高回报|高息|翻倍)[\s\S]{0,16}?"
            r"(?:保本|保收益|稳赚)"
        ),
    ),
    (
        "guaranteed return",
        _rx(r"guaranteed\s+(?:returns?|profit|income|yield)"),
    ),
    ("double your money", _rx(r"double\s+your\s+money")),
    # A return quoted per day is always bait ("每天收益5%", "5% a day", eval A08).
    (
        "每天收益%",
        _rx(r"(?:每天|每日|日)(?:的)?(?:收益|回报|利息|赚)?\s*\d+(?:\.\d+)?\s*[%％]"),
    ),
    (
        "% a day",
        _rx(
            r"\d+(?:\.\d+)?\s*[%％]\s*(?:a|per|every)\s*day"
            r"|(?<![a-z])daily\s+(?:returns?|profits?|interest)\s+of\s+\d"
        ),
    ),
    (
        "guaranteed %",
        _rx(r"guaranteed[\s\S]{0,24}?\d+(?:\.\d+)?\s*[%％]|\d+(?:\.\d+)?\s*[%％][\s\S]{0,24}?guaranteed"),
    ),
    (
        "risk-free high return",
        _rx(r"risk[\s-]*free\s+(?:high\s+)?(?:returns?|profit|yield)"),
    ),
)

_HEALTH_PRODUCT: tuple[_Phrase, ...] = (
    (
        "保健品治病",
        _rx(
            r"保健品[\s\S]{0,40}?"
            r"(?:治好|治愈|根治|治疗|降(?:血压|血糖|血脂)|调理好|包治)"
        ),
    ),
    ("包治百病", _rx(r"包治百病")),
    ("特效药", _rx(r"特效药")),
)

_PHISHING_LINK: tuple[_Phrase, ...] = (
    (
        "链接+账户提示",
        _rx(
            r"(?:https?://|www\.|bit\.ly|t\.cn|[\w-]+\.(?:cn|com)(?![a-z]))"
            r"[\s\S]{0,80}?"
            r"(?:账户异常|账户冻结|认证|验证|升级|过期|积分兑换|包裹|快递|"
            r"清关|关税|(?<![a-z])suspended(?![a-z])|(?<![a-z])verify(?![a-z])|"
            r"(?<![a-z])locked(?![a-z])|(?<![a-z])parcel(?![a-z])|"
            r"(?<![a-z])customs(?![a-z])|delivery\s*fee|points?\s*expire)"
        ),
    ),
    (
        "账户提示+链接",
        _rx(
            r"(?:账户异常|账户冻结|认证|验证|升级|过期|积分兑换|包裹|快递|"
            r"清关|关税|(?<![a-z])suspended(?![a-z])|(?<![a-z])verify(?![a-z])|"
            r"(?<![a-z])locked(?![a-z])|(?<![a-z])parcel(?![a-z])|"
            r"(?<![a-z])customs(?![a-z])|delivery\s*fee|points?\s*expire)"
            r"[\s\S]{0,80}?"
            r"(?:https?://|www\.|bit\.ly|t\.cn|[\w-]+\.(?:cn|com)(?![a-z]))"
        ),
    ),
)

_ADVANCE_FEE: tuple[_Phrase, ...] = (
    (
        "中奖+先付费",
        _rx(
            r"(?:中奖|奖金|大奖|补贴|退税|赔偿|(?<![a-z])prizes?(?![a-z])|"
            r"(?<![a-z])won(?![a-z'’])|(?<![a-z])winners?(?![a-z])|"
            r"(?<![a-z])lottery(?![a-z])|(?<![a-z])compensation(?![a-z])|"
            r"(?<![a-z])grants?(?![a-z]))"
            r"[\s\S]{0,80}?"
            r"(?:手续费|保证金|税费|激活费|解冻费|先交|先付|先转|"
            r"(?<![a-z])fees?(?![a-z])|(?<![a-z])deposits?(?![a-z])|"
            r"pay\s*first|processing\s*(?:charge|fee))"
        ),
    ),
    (
        "先付费+中奖",
        _rx(
            r"(?:手续费|保证金|税费|激活费|解冻费|先交|先付|先转|"
            r"(?<![a-z])fees?(?![a-z])|(?<![a-z])deposits?(?![a-z])|"
            r"pay\s*first|processing\s*(?:charge|fee))"
            r"[\s\S]{0,80}?"
            r"(?:中奖|奖金|大奖|补贴|退税|赔偿|(?<![a-z])prizes?(?![a-z])|"
            r"(?<![a-z])won(?![a-z'’])|(?<![a-z])winners?(?![a-z])|"
            r"(?<![a-z])lottery(?![a-z])|(?<![a-z])compensation(?![a-z])|"
            r"(?<![a-z])grants?(?![a-z]))"
        ),
    ),
)

_OFFICIAL_IMPERSONATION: tuple[_Phrase, ...] = (
    (
        "官方机构+异常",
        _rx(
            r"(?:医保|社保|养老金|医保局|社保局|人社局|卫健委|税务局|"
            r"medical\s*insurance|social\s*security|pension\s*office)"
            r"[\s\S]{0,60}?"
            r"(?:冒用|异常|冻结|停用|补缴|认证|核查|"
            r"联系[\s\S]{0,6}?(?:科长|主任|专员)|"
            r"(?<![a-z])suspended(?![a-z])|(?<![a-z])frozen(?![a-z])|"
            r"(?<![a-z])misused(?![a-z])|(?<![a-z])verify(?![a-z]))"
        ),
    ),
    (
        "异常+官方机构",
        _rx(
            r"(?:冒用|异常|冻结|停用|补缴|认证|核查|"
            r"联系[\s\S]{0,6}?(?:科长|主任|专员)|"
            r"(?<![a-z])suspended(?![a-z])|(?<![a-z])frozen(?![a-z])|"
            r"(?<![a-z])misused(?![a-z])|(?<![a-z])verify(?![a-z]))"
            r"[\s\S]{0,60}?"
            r"(?:医保|社保|养老金|医保局|社保局|人社局|卫健委|税务局|"
            r"medical\s*insurance|social\s*security|pension\s*office)"
        ),
    ),
)

_ELDER_INVESTMENT: tuple[_Phrase, ...] = (
    ("养老项目", _rx(r"养老项目")),
    ("养老公寓", _rx(r"养老公寓")),
    ("养老基金", _rx(r"养老基金")),
    ("以房养老", _rx(r"以房养老")),
    ("保本高息", _rx(r"保本高息")),
    ("稳赚不赔", _rx(r"稳赚不赔")),
    ("免费体检", _rx(r"免费体检")),
    ("免费旅游", _rx(r"免费旅游")),
    ("健康讲座", _rx(r"健康讲座")),
    ("收藏品升值", _rx(r"收藏品升值")),
    ("pension project", _rx(r"(?<![a-z])pension\s*project(?![a-z])")),
    ("guaranteed return", _rx(r"(?<![a-z])guaranteed\s*returns?(?![a-z])")),
    ("risk-free return", _rx(r"(?<![a-z])risk[\s-]*free\s*returns?(?![a-z])")),
    ("free health check", _rx(r"(?<![a-z])free\s*health\s*check(?![a-z])")),
)

_REMOTE_CONTROL: tuple[_Phrase, ...] = (
    (
        "下载远程软件",
        _rx(
            r"(?:下载|安装)[\s\S]{0,16}?"
            r"(?:向日葵|todesk|teamviewer|anydesk|屏幕共享|共享屏幕|"
            r"远程控制|会议软件)"
        ),
    ),
    (
        "install remote app",
        _rx(
            r"(?<![a-z])(?:install|download)\s+"
            r"(?:the\s+|an?\s+)?(?:anydesk|teamviewer)(?![a-z])"
        ),
    ),
    (
        "share your screen",
        _rx(
            r"(?<![a-z])share\s+(?:(?:your|ur|the)\s+)?screen(?![a-z])"
        ),
    ),
)

_INJECTED_INSTRUCTION: tuple[_Phrase, ...] = (
    # Only instructions aimed at the assistant's rules.  "忽略之前说的300" is a
    # customer correcting their own amount, and "备注写：10月房租转账" is an
    # ordinary memo.
    (
        "忽略规则",
        _rx(
            r"(?:忽略|无视|跳过|绕过)(?:以上|之前|上面|所有|全部)?的?"
            r"(?:规则|指令|指示|提示词?|系统|安全(?:规则|检查|验证)|限制|限额)"
        ),
    ),
    (
        "ignore instructions",
        _rx(
            r"(?<![a-z])(?:ignore|disregard|bypass)\s+"
            r"(?:(?:the|all|any|previous|prior|above|your|safety|security)\s+)*"
            r"(?:rules|instructions|prompt|system\s+prompt|checks|limits)(?![a-z])"
        ),
    ),
    (
        "备注指令",
        _rx(
            r"备注(?:写|填)(?:上)?[：:][\s\S]{0,20}?"
            r"(?:忽略|无视|系统|指令|规则|ignore|system|instruction)"
        ),
    ),
)

_MONEY_MOVE: tuple[_Pattern, ...] = (
    _rx(r"转(?!发|告|达|载|播)"),
    _rx(r"汇(?!报|率|总|编)"),
    _rx(r"打(?:钱|款|到|给|入|过去|过来)"),
    _rx(r"付(?:款|钱|账|到|给|进)"),
    _rx(r"存(?:进|入|到)"),
    _rx(r"交(?:钱|费)"),
    _rx(r"货款"),
    _rx(
        r"(?<![a-z])(?:transfer(?:s|red|ring)?|send(?:s|ing)?|sent|"
        r"pay(?:s|ing|ment)?|paid|wire(?:s|d)?|mov(?:e|es|ing|ed)|"
        r"remit(?:s|ted|ting)?|deposit(?:s|ed|ing)?)(?![a-z])"
    ),
    _rx(
        r"(?<![a-z])(?:zhuan(?:\s*zhang)?|da\s*qian|"
        r"hui\s*kuan|fu\s*kuan)(?![a-z])"
    ),
)

_OTP_VERB = (
    r"(?:告诉|说出|念|发给|发我|发送|报给|提供|发(?![现布生])|"
    r"(?<![a-z])(?:share|tell|send|give|read|forward|provide|"
    r"gao\s*su|shuo|fa\s*song|ti\s*gong|bao|nian|fa\s*gei)(?![a-z]))"
)
_OTP_CODE = (
    r"(?:验证码|动态码|短信码|校验码|短信(?:里|上)(?:的)?数字|"
    r"(?:收到|刚收到|短信)?的?(?:六|6)位(?:数字?|码)|"
    r"(?<![a-z])(?:the\s+)?(?:6|six)[\s-]*digits?(?![a-z])|"
    r"(?<![a-z])(?:number|digits)\s+(?:we|i|they)\s+(?:just\s+)?(?:texted|sent|messaged)(?![a-z])|"
    r"(?<![a-z])digits\s+(?:in|from)\s+the\s+(?:sms|text)(?![a-z])|"
    r"(?<![a-z])(?:code|otp)(?![a-z])|"
    r"(?<![a-z])yan\s*zheng\s*ma(?![a-z]))"
)
_OTP_PATTERNS: tuple[_Pattern, ...] = (
    _rx(rf"(?P<verb>{_OTP_VERB})[\s\S]{{0,20}}?(?P<code>{_OTP_CODE})"),
    _rx(rf"(?P<code>{_OTP_CODE})[\s\S]{{0,20}}?(?P<verb>{_OTP_VERB})"),
)

_ZH_BASE = "请立即停止转账和付款，不要按对方要求操作。"
_EN_BASE = "Stop any transfer or payment now and do not follow their instructions."
_ZH_HOTLINE = "如有疑问，请拨打 96110 反诈专线核实。"
_EN_HOTLINE = "If in doubt, call the 96110 anti-fraud hotline."

_ZH_ADVICE: dict[str, str] = {
    "safe_account": "公检法不会要求把钱转到“安全账户”，也不存在资金清查。",
    "police_claim": "接到自称警察、检察院或法院的电话，请先挂断，再通过官方号码回拨核实。",
    "impersonation": "亲友或供应商说换号、换账户时，请用原来的号码或当面核实。",
    "coached_secrecy": "骗子常要求你保密、不挂电话。真正的银行和警察不会这样做，请先告诉家人。",
    "injected_instruction": "有人让你照着输入的指令可能是诈骗，请不要照做。",
    "refund_bait": "正规退款不会索要验证码、银行卡密码，也不会让你先转账或下载软件。",
    "task_scam": "刷单、垫付、做任务返利都是诈骗，不要垫钱。",
    "loan_fee": "正规贷款在放款前不会收取保证金、手续费或解冻费。",
    "investment_bait": "“保本高收益”“稳赚不赔”都是骗局，不要投入资金。",
    "otp_request": "任何索要验证码的人都是骗子，验证码不能告诉任何人。",
    "family_card_gift": "不要把亲属卡/亲情卡送给陌生人或网友，这是常见骗局。",
    "remote_control": "任何让你下载远程控制或开屏幕共享的都是骗子，不要照做。",
    "health_product": "保健品不能治病，先交定金、包治百病都是常见骗局。",
    "phishing_link": "不要点短信或聊天里的链接，银行和快递不会用链接让你认证或补缴费用；请打开官方 App 核实。",
    "advance_fee": "中奖、补贴、赔偿要你先交钱的都是骗局，真的奖金不需要先付费。",
    "official_impersonation": "医保、社保、养老金部门不会打电话让你转账或提供验证码，请拨打 12393/12333 官方电话或到窗口核实。",
    "elder_investment": "养老项目、以房养老、保本高息、免费体检送礼品都是针对老人的常见骗局，投钱前先和家人商量。",
}

_EN_ADVICE: dict[str, str] = {
    "safe_account": "Police and courts never ask you to move money to a safe account.",
    "police_claim": "If someone claims to be police, a prosecutor or a court, hang up and call the official number back.",
    "impersonation": "If a relative or supplier says they changed number or account, verify using the old number.",
    "coached_secrecy": "Scammers ask you to keep it secret and stay on the line. Real banks and police never do; tell your family first.",
    "injected_instruction": "Instructions someone told you to type may be a scam; don't follow them.",
    "refund_bait": "A real refund never asks for a code or password, and never asks you to pay or download an app first.",
    "task_scam": "Paid tasks, advance payments and rebate jobs are scams; never pay first.",
    "loan_fee": "A real lender never charges a deposit, fee or unfreezing fee before releasing a loan.",
    "investment_bait": "Guaranteed high returns and risk-free profits are scams.",
    "otp_request": "Anyone who asks for your verification code is a scammer; never share it.",
    "family_card_gift": "Never gift a family card to strangers or online contacts; it is a common scam.",
    "remote_control": "Anyone asking you to install remote-control apps or share your screen is a scammer.",
    "health_product": "Health products cannot cure illness; 'pay a deposit first' and miracle cures are common scams.",
    "phishing_link": "Don't tap links in texts or chats; banks and couriers never ask you to verify or pay through a link. Check in the official app.",
    "advance_fee": "Any prize, grant or compensation that asks you to pay first is a scam; real payouts never cost money up front.",
    "official_impersonation": "Medical-insurance, social-security and pension offices never call to ask for transfers or codes. Check with the official hotline (12393 / 12333) or in person.",
    "elder_investment": "Pension projects, home-equity schemes, 'guaranteed returns' and free health checks with gifts are common scams aimed at older people; talk to family before paying.",
}


# -------------------------------------------------------------------- matching


def _entry_hits(
    entries: Iterable[_Phrase], forms: Iterable[str]
) -> list[str]:
    found: list[str] = []
    for label, pattern in entries:
        if label in found:
            continue
        for form in forms:
            if pattern.search(form):
                found.append(label)
                break
    return found


def _negated(form: str, start: int) -> bool:
    prefix = form[max(0, start - 12) : start]
    prefix = _BREAK_RE.split(prefix)[-1]
    return _NEGATION_RE.search(prefix) is not None


def _otp_hits(forms: Iterable[str]) -> list[str]:
    found: list[str] = []
    for form in forms:
        for pattern in _OTP_PATTERNS:
            for match in pattern.finditer(form):
                start = min(match.start("verb"), match.start("code"))
                if _negated(form, start):
                    continue
                code = match.group("code")
                label = "索要验证码" if _CJK_RE.search(code) else "OTP request"
                if label not in found:
                    found.append(label)
                break
    return found


_SAFETY_DETAIL_RE = _rx(
    r"(?:验证码|动态码|短信码|校验码|密码|"
    r"(?<![a-z])(?:code|otp|password)(?![a-z]))"
)


def _coached_hits(forms: Iterable[str]) -> list[str]:
    """Secrecy/coaching phrases, excluding ordinary code-safety advice.

    "不要告诉别人验证码" and "don't tell anyone your code" are warnings to the
    user, not a scammer coaching them, so a nearby code/password mention makes
    the secrecy phrase harmless.
    """

    found: list[str] = []
    for form in forms:
        if _SURPRISE_RE.search(form):
            continue
        for label, pattern in _COACHED_SECRECY:
            if label in found:
                continue
            for match in pattern.finditer(form):
                window = form[max(0, match.start() - 12) : match.end() + 24]
                if _SAFETY_DETAIL_RE.search(window):
                    continue
                found.append(label)
                break
    return found


def _has_money_move(forms: Iterable[str]) -> bool:
    return any(pattern.search(form) for form in forms for pattern in _MONEY_MOVE)


def _advice(flags: frozenset[str]) -> tuple[str, str]:
    if not flags:
        return (
            "未发现常见诈骗话术。不要向任何人透露验证码或密码；"
            + _ZH_HOTLINE,
            "No common scam script was detected. Never share your "
            "verification code or password. " + _EN_HOTLINE,
        )

    zh = [_ZH_BASE]
    en = [_EN_BASE]
    for flag in FLAG_ORDER:
        if flag in flags:
            zh.append(_ZH_ADVICE[flag])
            en.append(_EN_ADVICE[flag])
    zh.append(_ZH_HOTLINE)
    en.append(_EN_HOTLINE)
    return "".join(zh), " ".join(en)


# ---------------------------------------------------------------------- detect


def detect(
    text: str,
    *,
    recent_texts: Iterable[str] = (),
    amount_fen: int | None = None,
    payee_is_new: bool = False,
) -> ScamResult:
    """Scan raw user text (plus recent messages) for known scam scripts.

    The caller keeps the raw text: this function never rewrites or tokenizes
    it, so a phrase split across messages is still visible to the rules.
    """

    raw = [text, *recent_texts]
    normalized_parts: list[str] = []
    compact_parts: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        lowered = item.lower()
        normalized_parts.append(re.sub(r"\s+", " ", lowered).strip())
        compact_parts.append(re.sub(r"\s+", "", lowered))

    forms = ("\n".join(normalized_parts), "\n".join(compact_parts))

    safe_account = _entry_hits(_SAFE_ACCOUNT, forms)
    authority = _entry_hits(_AUTHORITY, forms)
    crime_claim = _entry_hits(_CRIME_CLAIM, forms)
    secrecy = _entry_hits(_SECRECY, forms)
    urgency = _entry_hits(_URGENCY, forms)
    impersonation = _entry_hits(_IMPERSONATION, forms)
    refund_bait = _entry_hits(_REFUND_BAIT, forms)
    task_scam = _entry_hits(_TASK_SCAM, forms)
    loan_fee = _entry_hits(_LOAN_FEE, forms)
    investment_bait = _entry_hits(_INVESTMENT_BAIT, forms)
    otp_request = _otp_hits(forms)
    family_card_gift = _entry_hits(_FAMILY_CARD_GIFT, forms)
    remote_control = _entry_hits(_REMOTE_CONTROL, forms)
    health_product = _entry_hits(_HEALTH_PRODUCT, forms)
    phishing_link = _entry_hits(_PHISHING_LINK, forms)
    advance_fee = _entry_hits(_ADVANCE_FEE, forms)
    official_impersonation = _entry_hits(_OFFICIAL_IMPERSONATION, forms)
    elder_investment = _entry_hits(_ELDER_INVESTMENT, forms)
    coached = _coached_hits(forms)
    injected_instruction = _entry_hits(_INJECTED_INSTRUCTION, forms)
    money_move = _has_money_move(forms)

    flags: set[str] = set()
    matches: list[tuple[str, str]] = []

    def add(flag: str, hits: Iterable[str]) -> None:
        hits = list(hits)
        if not hits:
            return
        flags.add(flag)
        for label in hits:
            pair = (flag, label)
            if pair not in matches:
                matches.append(pair)

    add("safe_account", safe_account)
    if money_move and (authority or crime_claim):
        add("police_claim", [*authority, *crime_claim])
    if money_move and impersonation:
        add("impersonation", impersonation)
    if coached and money_move:
        add("coached_secrecy", coached)
    add("injected_instruction", injected_instruction)
    add("refund_bait", refund_bait)
    add("task_scam", task_scam)
    add("loan_fee", loan_fee)
    add("investment_bait", investment_bait)
    add("otp_request", otp_request)
    add("family_card_gift", family_card_gift)
    add("remote_control", remote_control)
    if money_move and health_product:
        add("health_product", health_product)
    add("phishing_link", phishing_link)
    add("advance_fee", advance_fee)
    add("official_impersonation", official_impersonation)
    add("elder_investment", elder_investment)

    score = 0
    if flags:
        score = sum(
            _FLAG_SCORES[flag] for flag in FLAG_ORDER if flag in flags
        )
        if secrecy:
            score += _SECRECY_BONUS
        if urgency:
            score += _URGENCY_BONUS
        if payee_is_new:
            score += _NEW_PAYEE_BONUS
        if amount_fen is not None and amount_fen >= HIGH_AMOUNT_FEN:
            score += _HIGH_AMOUNT_BONUS
        score = min(score, 100)

    frozen = frozenset(flags)
    advice_zh, advice_en = _advice(frozen)
    return ScamResult(
        flags=frozen,
        score=score,
        matches=matches,
        advice_zh=advice_zh,
        advice_en=advice_en,
    )
