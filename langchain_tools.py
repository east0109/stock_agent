"""
LangChain Tools for Stock Analysis

This module provides LangChain-compatible tools for stock analysis,
with real tool execution and proper function calling.
"""
import logging
from typing import Dict, Any, List, Union
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

from data_fetcher import StockDataFetcher
from technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# Initialize components
data_fetcher = StockDataFetcher()
technical_indicators = TechnicalIndicators()

class FetchStockDataInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'RACE')")
    period: str = Field(description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")

class CalculateRSIInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate RSI for")
    period: int = Field(default=14, description="RSI calculation period")
    data_period: str = Field(default="1mo", description="Data period to fetch for RSI calculation")

class CalculateMovingAverageInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate moving average for")
    period: int = Field(default=20, description="Moving average period")
    data_period: str = Field(default="1mo", description="Data period to fetch for calculation")

class CalculateBollingerBandsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate Bollinger Bands for")
    period: int = Field(default=20, description="Moving average period")
    std_dev: float = Field(default=2.0, description="Standard deviation multiplier")
    data_period: str = Field(default="1mo", description="Data period to fetch for calculation")

class CalculateMACDInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate MACD for")
    fast_period: int = Field(default=12, description="Fast EMA period")
    slow_period: int = Field(default=26, description="Slow EMA period")
    signal_period: int = Field(default=9, description="Signal line period")
    data_period: str = Field(default="3mo", description="Data period to fetch for MACD calculation")

class CalculateAveragePriceInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate average price for")
    data_period: str = Field(default="1mo", description="Data period to fetch for calculation")

class FetchStockDataTool(BaseTool):
    name: str = "fetch_stock_data"
    description: str = "Fetch stock data for a given ticker and time period from Polygon.io"
    args_schema: type = FetchStockDataInput
    type: str = "function"

    def _run(self, ticker: str, period: str) -> str:
        """Fetch stock data."""
        try:
            logger.info(f"Fetching stock data for {ticker} over {period}")
            df = data_fetcher.fetch_stock_data(ticker, period)

            if len(df) > 0:
                latest_close = df['Close'].iloc[-1]
                date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"

                return f"Successfully fetched {len(df)} data points for {ticker} over {period}. Latest close: ${latest_close:.2f}. Date range: {date_range}"
            else:
                return f"No data found for {ticker} over {period}"

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return f"Error fetching stock data: {str(e)}"

    def run(self, *args, **kwargs) -> str:
        """Run the tool with flexible argument handling."""
        # Handle case where arguments might be passed as a single string
        if len(args) == 1 and isinstance(args[0], str):
            # Split the single argument into ticker and period
            parts = args[0].strip().split()
            if len(parts) >= 2:
                ticker = parts[0]
                period = parts[1]
            else:
                ticker = parts[0]
                period = "1mo"  # Default period
        elif len(args) >= 2:
            ticker = args[0]
            period = args[1]
        else:
            ticker = kwargs.get('ticker')
            period = kwargs.get('period', '1mo')

        if not ticker:
            return "Error: Ticker symbol is required"

        return self._run(ticker, period)



class CalculateRSITool(BaseTool):
    name: str = "calculate_rsi"
    description: str = "Calculate Relative Strength Index (RSI) for stock data. Fetches data first if needed."
    args_schema: type = CalculateRSIInput
    type: str = "function"

    def _run(self, ticker: str, period: int = 14, data_period: str = "1mo") -> str:
        """Calculate RSI."""
        try:
            logger.info(f"Calculating RSI for {ticker} with period {period}")

            # Fetch data first
            df = data_fetcher.fetch_stock_data(ticker, data_period)
            if len(df) == 0:
                return f"No data found for {ticker}"

            # Calculate RSI
            rsi_values = technical_indicators.calculate_rsi(df, period)

            if rsi_values:
                latest_rsi = rsi_values[-1]
                interpretation = self._interpret_rsi(latest_rsi)
                return f"RSI calculated for {ticker}: {len(rsi_values)} values, latest: {latest_rsi:.2f}. Interpretation: {interpretation}"
            else:
                return f"Insufficient data for RSI calculation. Need at least {period} data points."

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return f"Error calculating RSI: {str(e)}"

    def _interpret_rsi(self, rsi_value: float) -> str:
        """Interpret RSI value."""
        if rsi_value > 70:
            return "Overbought (potential sell signal)"
        elif rsi_value < 30:
            return "Oversold (potential buy signal)"
        else:
            return "Neutral"

class CalculateMovingAverageTool(BaseTool):
    name: str = "calculate_moving_average"
    description: str = "Calculate Simple Moving Average (SMA) for stock data. Fetches data first if needed."
    args_schema: type = CalculateMovingAverageInput
    type: str = "function"

    def _run(self, ticker: str, period: int = 20, data_period: str = "1mo") -> str:
        """Calculate moving average."""
        try:
            logger.info(f"Calculating moving average for {ticker} with period {period}")

            # Fetch data first
            df = data_fetcher.fetch_stock_data(ticker, data_period)
            if len(df) == 0:
                return f"No data found for {ticker}"

            # Calculate moving average
            ma_values = technical_indicators.calculate_moving_average(df, period)

            if ma_values:
                latest_ma = ma_values[-1]
                latest_close = df['Close'].iloc[-1]

                # Determine if price is above or below MA
                position = "above" if latest_close > latest_ma else "below"

                return f"Moving average for {ticker}: {period}-period MA is ${latest_ma:.2f}, current price is ${latest_close:.2f} ({position} MA)"
            else:
                return f"Insufficient data for moving average calculation. Need at least {period} data points."

        except Exception as e:
            logger.error(f"Error calculating moving average: {e}")
            return f"Error calculating moving average: {str(e)}"

class CalculateBollingerBandsTool(BaseTool):
    name: str = "calculate_bollinger_bands"
    description: str = "Calculate Bollinger Bands for stock data. Fetches data first if needed."
    args_schema: type = CalculateBollingerBandsInput
    type: str = "function"

    def _run(self, ticker: str, period: int = 20, std_dev: float = 2.0, data_period: str = "1mo") -> str:
        """Calculate Bollinger Bands."""
        try:
            logger.info(f"Calculating Bollinger Bands for {ticker} with period {period}")

            # Fetch data first
            df = data_fetcher.fetch_stock_data(ticker, data_period)
            if len(df) == 0:
                return f"No data found for {ticker}"

            # Calculate Bollinger Bands
            bb_result = technical_indicators.calculate_bollinger_bands(df, period, std_dev)

            if bb_result["upper_band"]:
                latest_close = df['Close'].iloc[-1]
                latest_upper = bb_result["upper_band"][-1]
                latest_middle = bb_result["middle_band"][-1]
                latest_lower = bb_result["lower_band"][-1]

                # Determine position relative to bands
                if latest_close > latest_upper:
                    position = "above upper band (overbought)"
                elif latest_close < latest_lower:
                    position = "below lower band (oversold)"
                else:
                    position = "within bands (normal)"

                return f"Bollinger Bands for {ticker}: Upper ${latest_upper:.2f}, Middle ${latest_middle:.2f}, Lower ${latest_lower:.2f}. Price is {position}."
            else:
                return f"Insufficient data for Bollinger Bands calculation. Need at least {period} data points."

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return f"Error calculating Bollinger Bands: {str(e)}"

class CalculateMACDTool(BaseTool):
    name: str = "calculate_macd"
    description: str = "Calculate MACD (Moving Average Convergence Divergence) for stock data. Fetches data first if needed."
    args_schema: type = CalculateMACDInput
    type: str = "function"

    def _run(self, ticker: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, data_period: str = "3mo") -> str:
        """Calculate MACD."""
        try:
            logger.info(f"Calculating MACD for {ticker}")

            # Fetch data first
            df = data_fetcher.fetch_stock_data(ticker, data_period)
            if len(df) == 0:
                return f"No data found for {ticker}"

            # Calculate MACD
            macd_result = technical_indicators.calculate_macd(df, fast_period, slow_period, signal_period)

            if macd_result["macd_line"]:
                latest_macd = macd_result["macd_line"][-1]
                latest_signal = macd_result["signal_line"][-1]
                latest_histogram = macd_result["histogram"][-1]

                # Determine MACD signal
                if latest_macd > latest_signal:
                    signal = "bullish (MACD above signal line)"
                else:
                    signal = "bearish (MACD below signal line)"

                return f"MACD for {ticker}: MACD {latest_macd:.4f}, Signal {latest_signal:.4f}, Histogram {latest_histogram:.4f}. Signal is {signal}."
            else:
                return f"Insufficient data for MACD calculation. Need at least {slow_period} data points."

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return f"Error calculating MACD: {str(e)}"

class CalculateAveragePriceTool(BaseTool):
    name: str = "calculate_average_price"
    description: str = "Calculate the average closing price for all available data points. Fetches data first if needed."
    args_schema: type = CalculateAveragePriceInput
    type: str = "function"

    def _run(self, ticker: str, data_period: str = "1mo") -> str:
        """Calculate average price."""
        try:
            logger.info(f"Calculating average price for {ticker}")

            # Fetch data first
            df = data_fetcher.fetch_stock_data(ticker, data_period)
            if len(df) == 0:
                return f"No data found for {ticker}"

            # Calculate average price
            avg_price = technical_indicators.calculate_average_price(df)
            latest_close = df['Close'].iloc[-1]

            # Determine if current price is above or below average
            position = "above" if latest_close > avg_price else "below"
            difference = abs(latest_close - avg_price)
            percentage = (difference / avg_price) * 100

            return f"Average price for {ticker} over {data_period}: ${avg_price:.2f}. Current price ${latest_close:.2f} is {position} average by ${difference:.2f} ({percentage:.1f}%)."

        except Exception as e:
            logger.error(f"Error calculating average price: {e}")
            return f"Error calculating average price: {str(e)}"

def get_all_tools():
    """Get all available LangChain tools."""
    return [
        FetchStockDataTool(),
        CalculateRSITool(),
        CalculateMovingAverageTool(),
        CalculateBollingerBandsTool(),
        CalculateMACDTool(),
        CalculateAveragePriceTool()
    ]
