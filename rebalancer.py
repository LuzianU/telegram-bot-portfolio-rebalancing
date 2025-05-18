from dataclasses import asdict, dataclass
import datetime
from dateutil.relativedelta import relativedelta
import json
import yfinance as yf
from enum import Enum
import logging

logging.getLogger("yfinance").disabled = True
logging.getLogger("yfinance").propagate = False
logging.getLogger("peewee").disabled = True
logging.getLogger("peewee").propagate = False

logger = logging.getLogger(__name__)


def check_ticker_available(ticker: str) -> bool:
    """
    Checks if a ticker is available via the Yahoo Finance API.
    """
    try:
        info = yf.Ticker(ticker).history(period="1mo", interval="1d")
        return len(info) > 0
    except Exception:
        return False


def get_current_price(ticker: str) -> float:
    """
    Gets the current price of a ticker.
    """
    info = yf.Ticker(ticker).history(period="1mo", interval="1d")
    return info["Close"].iloc[-1].item()


@dataclass
class SavingsPlan:
    class Frequency(Enum):
        Monthly = 1

    amount: float
    frequency: Frequency

    def __repr__(self):
        return json.dumps(asdict(self), indent=4, default=str)


@dataclass
class Ticker:
    ticker: str
    alias: str
    allocation: float
    savings_plan: SavingsPlan
    holdings: float

    def __init__(
        self,
        ticker: str,
        alias: str,
        allocation: float,
        savings_plan: SavingsPlan,
        holdings: float,
    ):
        self.ticker = ticker
        self.alias = alias
        self.allocation = allocation
        self.savings_plan = savings_plan
        self.holdings = holdings

    def get_price(self, date: datetime.date):
        interval = "1m" if date > datetime.date.today() - relativedelta(days=30) else "1d"
        data = yf.download(
            self.ticker,
            start=date,
            end=date + relativedelta(days=1),
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if data is None:
            return None
        row = data[data.index.date == date]  # type: ignore
        if len(row) == 0:
            return None
        open = row["Open"].iloc[0].item()
        close = row["Close"].iloc[-1].item()
        avg_close = row["Close"].mean().item()
        return {
            "Open": open,
            "Close": close,
            "Avg Close": avg_close,
        }

    def __repr__(self):
        return json.dumps(asdict(self), indent=4, default=str)


@dataclass
class Event:
    date: datetime.date
    ticker: Ticker
    amount: float
    shares: float


@dataclass
class SavingsPlanEvent(Event):
    pass


@dataclass
class RebalanceEvent(Event):
    needs_rebalance: bool


@dataclass
class Portfolio:
    name: str
    tickers: list[Ticker]
    absolute_band: float = 0.05
    relative_band: float = 0.25
    last_savings_plan_month: int = 0

    def __init__(
        self,
        name: str,
        tickers: list[Ticker],
        absolute_band: float = 0.05,
        relative_band: float = 0.25,
        last_savings_plan_month: int = 0,
    ):
        if abs(sum(t.allocation for t in tickers) - 1.0) > 1e-9:
            logger.error(f"Tickers for portfolio {name} do not sum to 1.0")
            raise ValueError(f"Tickers for portfolio {name} do not sum to 1.0")
        self.name = name
        self.tickers = tickers
        self.absolute_band = absolute_band
        self.relative_band = relative_band
        self.last_savings_plan_month = last_savings_plan_month

    def rebalance(self, rebalance_events: list[RebalanceEvent]):
        for event in rebalance_events:
            event.ticker.holdings += event.shares

            operation = "Bought" if event.shares > 0 else "Sold"
            logger.info(
                f"{event.date}: {operation} {event.amount} of {event.ticker.ticker} ({event.ticker.holdings} shares)"
            )

    def update(self, date: datetime.date) -> list[Event]:
        events = []

        prices = [t.get_price(date) for t in self.tickers]
        if any(p is None for p in prices):
            logger.info(f"{date}: No price data found for tickers: {prices}")
            return events

        # Manage savings plan
        if date.month != self.last_savings_plan_month:
            self.last_savings_plan_month = date.month

            for ticker in self.tickers:
                if ticker.savings_plan.amount == 0:
                    continue
                assert (
                    ticker.savings_plan.frequency == SavingsPlan.Frequency.Monthly
                ), "Savings plan frequency must be monthly"
                price = ticker.get_price(date)["Avg Close"]  # type: ignore
                shares_to_buy = ticker.savings_plan.amount / price
                ticker.holdings += shares_to_buy
                logger.info(
                    f"{date}: Bought {ticker.savings_plan.amount} of {ticker.ticker} ({ticker.holdings} shares)"
                )
                events.append(SavingsPlanEvent(date, ticker, ticker.savings_plan.amount, shares_to_buy))

        current_abs_allocs = [price["Close"] * t.holdings for t, price in zip(self.tickers, prices)]  # type: ignore
        current_wealth = sum(current_abs_allocs)
        current_per_allocs = [c / current_wealth for c in current_abs_allocs]

        logger.info(f"{date}: Current absolute allocation: {current_abs_allocs}")
        logger.info(f"{date}: Current relative allocation: {current_per_allocs}")

        needs_rebalance = any(
            (abs(current_per_alloc - ticker.allocation) > ticker.allocation * self.relative_band)
            or (abs(current_per_alloc - ticker.allocation) > self.absolute_band)
            for ticker, current_per_alloc in zip(self.tickers, current_per_allocs)
        )

        desired_amount_per_ticker = [(t.allocation * current_wealth) for t in self.tickers]

        # Sell all tickers that are over their allocation
        for ticker, desired_amount, price in zip(self.tickers, desired_amount_per_ticker, prices):
            if desired_amount < price["Close"] * ticker.holdings:  # type: ignore
                amount_to_sell = price["Close"] * ticker.holdings - desired_amount  # type: ignore
                shares_to_sell = amount_to_sell / price["Close"]  # type: ignore
                events.append(RebalanceEvent(date, ticker, -amount_to_sell, -shares_to_sell, needs_rebalance))

        # Buy all tickers that are under their allocation
        for ticker, desired_amount, price in zip(self.tickers, desired_amount_per_ticker, prices):
            if desired_amount > price["Close"] * ticker.holdings:  # type: ignore
                amount_to_buy = desired_amount - price["Close"] * ticker.holdings  # type: ignore
                shares_to_buy = amount_to_buy / price["Close"]  # type: ignore
                events.append(RebalanceEvent(date, ticker, amount_to_buy, shares_to_buy, needs_rebalance))

        if needs_rebalance:
            logger.info(f"{date}: Rebalance needed: Current allocation: {current_abs_allocs}")

            # Recalculate current allocations
            current_abs_allocs = [price["Close"] * t.holdings for t, price in zip(self.tickers, prices)]  # type: ignore
            logger.debug(f"{date}: Absolute allocations after rebalance: {current_abs_allocs}")
        else:
            logger.debug(f"{date}: All tickers are in band: Current allocation: {current_abs_allocs}")

        return events

    def __repr__(self):
        return json.dumps(asdict(self), indent=4, default=str)
