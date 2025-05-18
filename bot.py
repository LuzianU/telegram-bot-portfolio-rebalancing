from dataclasses import dataclass
import datetime
import json
import logging
import os
import threading
import rebalancer
from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    BotCommand,
    CallbackQuery,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
import logging.handlers

# --- Logging Setup Start ---
# Get root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
root_logger.addHandler(ch)

# Determine the root directory for the log file (directory of bot.py)
root_dir = os.path.dirname(os.path.abspath(__file__))

# Create file handler
fh = logging.FileHandler(os.path.join(root_dir, "app.log"))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
root_logger.addHandler(fh)
# --- Logging Setup End ---

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class AssetInfo:
    ticker: str
    alias: str
    allocation: float
    shares: float
    monthly_savings: float
    rebalance_delta_amount: float
    rebalance_delta_shares: float

    def __init__(
        self,
        ticker: str,
        alias: str,
        allocation: float,
        shares: float,
        monthly_savings: float,
        rebalance_delta_amount: float = 0,
        rebalance_delta_shares: float = 0,
    ):
        self.ticker = ticker
        self.alias = alias
        self.allocation = allocation
        self.shares = shares
        self.monthly_savings = monthly_savings
        self.rebalance_delta_amount = rebalance_delta_amount
        self.rebalance_delta_shares = rebalance_delta_shares

        if not rebalancer.check_ticker_available(ticker):
            raise ValueError(f"Ticker {ticker} is not available")

        if self.allocation < 0 or self.allocation > 1:
            raise ValueError(f"Allocation {self.allocation} is not valid")

        if self.shares < 0:
            raise ValueError(f"Shares {self.shares} is not valid")

        if self.monthly_savings < 0:
            raise ValueError(f"Monthly savings {self.monthly_savings} is not valid")

    @classmethod
    def from_string(cls, string: str) -> "AssetInfo":
        try:
            ticker, alias, allocation, shares, monthly_savings = string.split(",")
            return cls(ticker.strip(), alias.strip(), float(allocation), float(shares), float(monthly_savings))
        except Exception as e:
            raise ValueError(f"Invalid asset string: {string}\nError: {str(e)}") from e

    def to_string(self) -> str:
        return f"{self.ticker},{self.alias},{self.allocation},{self.shares},{self.monthly_savings}"

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "alias": self.alias,
            "allocation": self.allocation,
            "shares": self.shares,
            "monthly_savings": self.monthly_savings,
            "rebalance_delta_amount": self.rebalance_delta_amount,
            "rebalance_delta_shares": self.rebalance_delta_shares,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AssetInfo":
        return cls(
            data["ticker"],
            data["alias"],
            data["allocation"],
            data["shares"],
            data["monthly_savings"],
            data["rebalance_delta_amount"] if "rebalance_delta_amount" in data else 0,
            data["rebalance_delta_shares"] if "rebalance_delta_shares" in data else 0,
        )


class PortfolioInfo:
    name: str
    last_savings_plan_month: int
    assets: list[AssetInfo]

    def __init__(self, name: str, last_savings_plan_month: int, assets: list[AssetInfo]):
        self.name = name
        self.last_savings_plan_month = last_savings_plan_month
        self.assets = assets

    @classmethod
    def from_string(cls, string: str) -> "PortfolioInfo":
        try:
            name, last_savings_plan_month, *assets = string.split(",")
            assets_string = ",".join(assets).strip()
            last_savings_plan_month = int(last_savings_plan_month)
            assets = [AssetInfo.from_string(asset) for asset in assets_string.split(";")]
            if len(assets) == 0:
                raise ValueError(f"No assets found in portfolio {name}")
            if abs(sum(asset.allocation for asset in assets) - 1) > 1e-6:
                raise ValueError(
                    f"Allocation sum in portfolio {name} is not 1: {sum(asset.allocation for asset in assets)}"
                )
            return cls(name.strip(), last_savings_plan_month, assets)
        except Exception as e:
            raise ValueError(f"Invalid portfolio string: {string}\nError: {str(e)}") from e

    def to_string(self) -> str:
        return f"{self.name},{self.last_savings_plan_month},{';'.join(asset.to_string() for asset in self.assets)}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "last_savings_plan_month": self.last_savings_plan_month,
            "assets": [asset.to_dict() for asset in self.assets],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioInfo":
        assets = [AssetInfo.from_dict(asset_data) for asset_data in data["assets"]]
        if len(assets) != len(set(asset.ticker for asset in assets)):
            raise ValueError("Assets must have unique tickers")
        return cls(data["name"], data["last_savings_plan_month"], assets)


class UserData:
    portfolios: list[PortfolioInfo]

    def __init__(self, portfolios: list[PortfolioInfo]):
        self.portfolios = portfolios

    @classmethod
    def from_string(cls, string: str) -> "UserData":
        try:
            portfolios = [
                PortfolioInfo.from_string(portfolio.strip()) for portfolio in string.splitlines() if portfolio.strip()
            ]
            if len(portfolios) == 0:
                raise ValueError("No portfolios found")
            return cls(portfolios)
        except Exception as e:
            raise ValueError(f"{e}") from e

    def to_dict(self) -> dict:
        return {"portfolios": [portfolio.to_dict() for portfolio in self.portfolios]}

    @classmethod
    def from_dict(cls, data: dict) -> "UserData":
        portfolios = [PortfolioInfo.from_dict(p_data) for p_data in data["portfolios"]]
        if len(portfolios) != len(set(portfolio.name for portfolio in portfolios)):
            raise ValueError("Portfolios must have unique names")
        return cls(portfolios)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4, default=str)


class Database:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super(Database, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, path: str = os.path.join(root_dir, "db.json")):
        if self._initialized:
            return
        self._initialized = True
        self.path = path
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("{}")

    def get_all_user_ids(self) -> list[int]:
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(data.keys() - ["last_update_date"])

    def get_last_update_date(self) -> datetime.date:
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return datetime.date.fromisoformat(data.get("last_update_date", "0001-01-01"))

    def set_last_update_date(self, date: datetime.date):
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["last_update_date"] = date.isoformat()
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

    def get_data(self, user_id: int) -> UserData | None:
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            user_data_dict = data.get(str(user_id), None)
            if user_data_dict:
                return UserData.from_dict(user_data_dict)
            return None

    def set_data(self, user_id: int, data: UserData):
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            all_data[str(user_id)] = data.to_dict()
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)


def simulate_date(
    portfolio_info: PortfolioInfo, date: datetime.date, last_savings_plan_month: int
) -> list[rebalancer.Event]:
    logger.info(
        f"Simulating date {date} for portfolio {portfolio_info.name} with last savings plan month {last_savings_plan_month}"
    )

    tickers = []
    for asset in portfolio_info.assets:
        savings_plan = rebalancer.SavingsPlan(asset.monthly_savings, rebalancer.SavingsPlan.Frequency.Monthly)
        tickers.append(rebalancer.Ticker(asset.ticker, asset.alias, asset.allocation, savings_plan, asset.shares))

    portfolio = rebalancer.Portfolio(portfolio_info.name, tickers, last_savings_plan_month=last_savings_plan_month)
    events = portfolio.update(date)

    return events


BOT_COMMANDS = [
    BotCommand("help", "Get help with the bot"),
    BotCommand("status", "Get the status of your portfolios"),
    BotCommand("set", "Set your portfolios"),
    BotCommand("update", "Force an update of your portfolios (done daily)"),
    BotCommand("rebalance", "Do this to rebalance your portfolios based on the latest update's values"),
    BotCommand("portfolios", "Get the representation of your portfolios"),
]


async def bot_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    assert update.message is not None

    text = "Available commands:\n\n"
    for command in BOT_COMMANDS:
        text += f"/{command.command} — {command.description}\n"

    text += "\n"
    text += "The /set command expects the following format:\n\n"
    text += "<code>Portfolio name, Last executed saving plan month (0 for automatic), Asset 1, Asset 2, ...</code>\n"
    text += "<b>Asset format:</b> <code>Ticker, Alias, Allocation, Shares owned, Monthly saving amount</code>\n\n"
    text += "Example:\n"
    text += '<pre language="csv">Portfolio 1, 0, SXR8.DE, S&P 500, 0.8, 50, 160; 4GLD.DE, Gold, 0.2, 20, 40\n'
    text += "Portfolio 2, 0, GOOGL, Google Stock, 0.34, 50, 100; NVDA, Nvidia Stock, 0.33, 50, 100; AAPL, Apple Stock, 0.33, 50, 100</pre>\n\n"
    text += "Ticker data is taken from <a href='https://finance.yahoo.com/'>Yahoo Finance</a>."

    await update.message.reply_text(text, parse_mode="HTML", disable_web_page_preview=True)


async def bot_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        assert update.message is not None
        assert update.message.from_user is not None

        db = Database()
        user_id = update.message.from_user.id
        data = db.get_data(user_id)
        if data is None:
            await update.message.reply_text(f"You don't have any portfolios set up yet.")
            return
        last_update_date = db.get_last_update_date()
        days_since_last_update = (datetime.date.today() - last_update_date).days
        days_since_last_update_str = "Today"
        if days_since_last_update == 1:
            days_since_last_update_str = "Yesterday"
        elif days_since_last_update > 1:
            days_since_last_update_str = f"{days_since_last_update} days ago"
        text = f"_Last update: {last_update_date} ({days_since_last_update_str})_\n\n"
        portfolio_values = []
        for i, portfolio in enumerate(data.portfolios):
            text += f"📂 *{portfolio.name}:* $$${i}$$$\n"
            portfolio_value = []
            for j, asset in enumerate(portfolio.assets):
                current_price = rebalancer.get_current_price(asset.ticker)
                current_value = asset.shares * current_price
                text += f"    ┬ *{asset.alias}* ({asset.allocation * 100:.2f}%)\n"
                text += f"    ├ 💶 {current_value:.2f}€ $$${i}${j}$$$\n"
                text += f"    ├ 📊 {asset.shares:.2f} shares\n"
                text += f"    └ 🔄 {asset.monthly_savings:.2f}€ per month\n"
                portfolio_value.append(current_value)
            portfolio_values.append(portfolio_value)

            if i < len(data.portfolios) - 1:
                text += "\n"

        for i, (portfolio, portfolio_value) in enumerate(zip(data.portfolios, portfolio_values)):
            portfolio_value_sum = sum(portfolio_value)
            text = text.replace(f"$$${i}$$$", f"{portfolio_value_sum:.2f}€")

            for j, (asset, value) in enumerate(zip(portfolio.assets, portfolio_value)):
                text = text.replace(f"$$${i}${j}$$$", f"({value/portfolio_value_sum*100:.2f}%)")

        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")  # type: ignore


async def bot_portfolios(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        assert update.message is not None
        assert update.message.from_user is not None

        db = Database()
        user_id = update.message.from_user.id
        data = db.get_data(user_id)
        if data is None:
            await update.message.reply_text(f"You don't have any portfolios set up yet.")
            return

        text = "\n".join(portfolio.to_string() for portfolio in data.portfolios)
        text = '<pre language="csv">' + text + "</pre>"
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")  # type: ignore


async def bot_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        assert update.message is not None
        assert update.message.from_user is not None
        assert context.user_data is not None

        await update.message.reply_text("Please send the portfolio string.")

        context.user_data["wait_for_portfolio_string"] = True
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")  # type: ignore


async def bot_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        assert update.message is not None
        assert update.message.from_user is not None
        assert context.user_data is not None

        if context.user_data.get("wait_for_portfolio_string", False):
            context.user_data["wait_for_portfolio_string"] = False
            text = update.message.text
            text = text if text else ""
            try:
                data = UserData.from_string(text)
            except Exception as e:
                await update.message.reply_text(f"Error: {str(e)}")
                return

            db = Database()
            user_id = update.message.from_user.id
            db.set_data(user_id, data)

            await update.message.reply_text("Portfolios set successfully.")

        if context.user_data.get("wait_for_rebalance_string", False):
            context.user_data["wait_for_rebalance_string"] = False
            text = update.message.text
            text = text.strip() if text else ""

            # Get the portfolio from the database based on the text
            db = Database()
            user_id = update.message.from_user.id
            data = db.get_data(user_id)
            if data is None:
                await update.message.reply_text(f"You don't have any portfolios set up yet.")
                return

            # Get the index of the portfolio
            portfolio_index = next((i for i, p in enumerate(data.portfolios) if p.name == text), None)
            if portfolio_index is None:
                await update.message.reply_text(f"Portfolio {text} not found.")
                return

            async def send_message(user_id: int, text: str, parse_mode: str = "Markdown"):
                assert update.message is not None
                await update.message.reply_text(text, parse_mode=parse_mode)

            await logic_rebalance(portfolio_index, send_message, user_id)

    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")  # type: ignore


async def bot_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        assert update.message is not None
        assert update.message.from_user is not None
        assert context.user_data is not None

        async def send_message(user_id: int, text: str, parse_mode: str = "Markdown"):
            assert update.message is not None
            await update.message.reply_text(text, parse_mode=parse_mode)

        update_date = datetime.date.today()
        await logic_update(update_date, send_message, update.message.from_user.id)

        db = Database()
        db.set_last_update_date(update_date)

        await update.message.reply_text(f"Portfolios updated successfully.")

    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")  # type: ignore


async def bot_rebalance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        assert update.message is not None
        assert update.message.from_user is not None
        assert context.user_data is not None

        db = Database()
        user_id = update.message.from_user.id
        data = db.get_data(user_id)
        if data is None:
            await update.message.reply_text(f"You don't have any portfolios set up yet.")
            return

        text = "Please select the portfolio to rebalance:\n"
        text += "\n".join(f"- `{p.name}`" for p in data.portfolios)
        await update.message.reply_text(text, parse_mode="Markdown")

        context.user_data["wait_for_rebalance_string"] = True
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")  # type: ignore


async def logic_rebalance(portfolio_index: int, send_message, user_id: int):
    db = Database()
    data = db.get_data(user_id)
    if data is None:
        return

    logger.info(f"Rebalancing portfolio {data.portfolios[portfolio_index].name} for user {user_id}")

    deltas = []
    for i, _ in enumerate(data.portfolios[portfolio_index].assets):
        data.portfolios[portfolio_index].assets[i].shares += (
            data.portfolios[portfolio_index].assets[i].rebalance_delta_shares
        )

        deltas.append(
            (
                data.portfolios[portfolio_index].assets[i].alias,
                data.portfolios[portfolio_index].assets[i].rebalance_delta_amount,
                data.portfolios[portfolio_index].assets[i].rebalance_delta_shares,
            )
        )

        data.portfolios[portfolio_index].assets[i].rebalance_delta_amount = 0
        data.portfolios[portfolio_index].assets[i].rebalance_delta_shares = 0

    text = f"⚖️ *Executed rebalance for {data.portfolios[portfolio_index].name}*\n"
    deltas = sorted(deltas, key=lambda x: x[1])
    for alias, delta_amount, delta_shares in deltas:
        prefix = "🟢 Bought" if delta_amount > 0 else "🔴 Sold"
        text += f"{prefix} {abs(delta_amount):.2f}€ ({abs(delta_shares):.2f} shares) of {alias}\n"
    await send_message(user_id, text, parse_mode="Markdown")

    db.set_data(user_id, data)


async def logic_update(update_date: datetime.date, send_message, specific_user_id: int | None = None):
    db = Database()

    user_ids = db.get_all_user_ids() if specific_user_id is None else [specific_user_id]

    for user_id in user_ids:
        logger.info(f"Updating portfolios for user {user_id}")
        data = db.get_data(user_id)
        if data is None:
            continue
        data_modified = False

        for i, portfolio in enumerate(data.portfolios):
            events = simulate_date(portfolio, update_date, portfolio.last_savings_plan_month)

            savings_plan_events = [event for event in events if isinstance(event, rebalancer.SavingsPlanEvent)]
            rebalance_events = [
                event for event in events if isinstance(event, rebalancer.RebalanceEvent) if event.needs_rebalance
            ]

            if len(events) > 0:
                data_modified = True

                if len(savings_plan_events) > 0:
                    data.portfolios[i].last_savings_plan_month = update_date.month

                for j, asset in enumerate(portfolio.assets):
                    for event in events:
                        if event.ticker.ticker == asset.ticker:
                            data.portfolios[i].assets[j].rebalance_delta_amount = event.amount
                            data.portfolios[i].assets[j].rebalance_delta_shares = event.shares

            if len(savings_plan_events) > 0:
                text = f"🔄 *Saving plan executed for {portfolio.name}*\n"
                for event in savings_plan_events:
                    text += f"🟢 Bought {event.amount:.2f}€ ({event.shares:.2f} shares) of {event.ticker.alias}\n"
                await send_message(user_id, text, parse_mode="Markdown")
            if len(rebalance_events) > 0:
                text = f"⚖️ *Rebalance needed for {portfolio.name}*\n"
                for event in rebalance_events:
                    prefix = "🟢 Buy" if event.amount >= 0 else "🔴 Sell"
                    text += (
                        f"{prefix} {abs(event.amount):.2f}€ ({abs(event.shares):.2f} shares) of {event.ticker.alias}\n"
                    )
                await send_message(user_id, text, parse_mode="Markdown")

        if data_modified:
            db.set_data(user_id, data)

        logger.info(f"Portfolios updated successfully for user {user_id}")


async def bot_post_init(application: Application):
    try:
        await application.bot.set_my_commands(BOT_COMMANDS)

        db = Database()
        last_update_date = db.get_last_update_date()
        # Update on yesterday to get market close prices
        update_date = datetime.date.today() - datetime.timedelta(days=1)

        if last_update_date >= update_date:
            logger.info(
                f"Last update date {last_update_date} is greater than or equal to update date {update_date}, skipping update"
            )
            return

        await logic_update(update_date, application.bot.send_message)

        db.set_last_update_date(update_date)
        logger.info(f"Last update date set to {update_date}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def main():
    load_dotenv()
    token = os.getenv("TOKEN")
    assert token, "TOKEN is not set"

    app = Application.builder().token(token).post_init(bot_post_init).build()
    app.add_handler(CommandHandler("start", bot_help))
    app.add_handler(CommandHandler("help", bot_help))
    app.add_handler(CommandHandler("status", bot_status))
    app.add_handler(CommandHandler("portfolios", bot_portfolios))
    app.add_handler(CommandHandler("set", bot_set))
    app.add_handler(CommandHandler("update", bot_update))
    app.add_handler(CommandHandler("rebalance", bot_rebalance))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
