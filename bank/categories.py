"""Spending categories and merchant -> category lookup."""

CATEGORIES: tuple[str, ...] = (
    "food",
    "groceries",
    "transport",
    "shopping",
    "bills_utilities",
    "housing",
    "entertainment",
    "subscriptions",
    "health",
    "education",
    "transfer",
    "income",
    "other",
)

# Every merchant and subscription name in data_spec/customer_segments.yaml.
MERCHANT_CATEGORY: dict[str, str] = {
    # merchants
    "美团": "food",
    "饿了么": "food",
    "滴滴": "transport",
    "盒马": "groceries",
    "京东": "shopping",
    "瑞幸": "food",
    "淘宝": "shopping",
    "拼多多": "shopping",
    "食堂": "food",
    "加油站": "transport",
    "电动车维修": "other",
    "便利店": "food",
    "菜市场": "groceries",
    "药店": "health",
    "社区医院": "health",
    "超市": "groceries",
    "幼儿园": "education",
    "儿童医院": "health",
    "菜批发市场": "groceries",
    "燃气公司": "bills_utilities",
    "美团商家后台": "other",
    "Starbucks": "food",
    # subscriptions
    "爱奇艺 VIP": "entertainment",
    "网易云音乐": "entertainment",
    "QQ音乐": "entertainment",
    "Keep": "health",
    "百度网盘": "subscriptions",
    "iCloud": "subscriptions",
    "B站大会员": "entertainment",
    "腾讯视频": "entertainment",
    "王者荣耀 月卡": "entertainment",
    "夸克网盘": "subscriptions",
    "老年手机 套餐": "bills_utilities",
    "广场舞 会员": "entertainment",
    "少儿英语 App": "education",
    "儿童绘本 会员": "education",
    "家庭宽带": "bills_utilities",
    "Netflix via VPN card": "entertainment",
    "Spotify": "entertainment",
    # fallback plan used for personas that declare no subscriptions
    "中国移动 套餐": "bills_utilities",
}


def category_for(name: str) -> str:
    """Return the category for a merchant/subscription name."""
    return MERCHANT_CATEGORY.get(name, "other")
