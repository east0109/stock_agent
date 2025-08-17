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
    data_period: str = Field(description="Data period to fetch for RSI calculation (e.g., 1d, 5d, 1mo, 3mo)")

class CalculateMovingAverageInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate moving average for")
    period: int = Field(default=20, description="Moving average period")
    data_period: str = Field(description="Data period to fetch for calculation (e.g., 1d, 5d, 1mo, 3mo)")

class CalculateBollingerBandsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate Bollinger Bands for")
    period: int = Field(default=20, description="Moving average period")
    std_dev: float = Field(default=2.0, description="Standard deviation multiplier")
    data_period: str = Field(description="Data period to fetch for calculation (e.g., 1d, 5d, 1mo, 3mo)")

class CalculateMACDInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate MACD for")
    fast_period: int = Field(default=12, description="Fast EMA period")
    slow_period: int = Field(default=26, description="Slow EMA period")
    signal_period: int = Field(default=9, description="Signal line period")
    data_period: str = Field(description="Data period to fetch for MACD calculation (e.g., 1mo, 3mo, 6mo)")

class CalculateAveragePriceInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to calculate average price for")
    data_period: str = Field(description="Data period to fetch for calculation (e.g., 1d, 5d, 1mo, 3mo)")

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
            # First try to split by pipes (LangChain format), then colons, then commas, then spaces
            arg_string = args[0].strip()

            # Try pipe separation first (LangChain format)
            if '|' in arg_string:
                parts = [part.strip() for part in arg_string.split('|')]
            elif ':' in arg_string:
                parts = [part.strip() for part in arg_string.split(':')]
            elif ',' in arg_string:
                parts = [part.strip() for part in arg_string.split(',')]
            else:
                # Fall back to space separation
                parts = arg_string.split()

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

    def _run(self, ticker: str, period: int = 14, data_period: str = None) -> str:
        """Calculate RSI."""
        try:
            # Use provided data_period or default to 1mo if not specified
            if data_period is None:
                data_period = "1mo"

            logger.info(f"Calculating RSI for {ticker} with period {period} over {data_period}")

            # Fetch data first
            df = data_fetcher.fetch_stock_data(ticker, data_period)
            if len(df) == 0:
                return f"No data found for {ticker}"

            # Calculate RSI
            rsi_values = technical_indicators.calculate_rsi(df, period)

            if rsi_values:
                latest_rsi = rsi_values[-1]
                interpretation = self._interpret_rsi(latest_rsi)
                return f"RSI calculated for {ticker} over {data_period}: {len(rsi_values)} values, latest: {latest_rsi:.2f}. Interpretation: {interpretation}"
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

    def run(self, *args, **kwargs) -> str:
        """Run the tool with flexible argument handling."""
        # Handle case where arguments might be passed as a single string
        if len(args) == 1 and isinstance(args[0], str):
            # First try to split by pipes (LangChain format), then colons, then commas, then spaces
            arg_string = args[0].strip()

            # Try pipe separation first (LangChain format)
            if '|' in arg_string:
                parts = [part.strip() for part in arg_string.split('|')]
            elif ':' in arg_string:
                parts = [part.strip() for part in arg_string.split(':')]
            elif ',' in arg_string:
                parts = [part.strip() for part in arg_string.split(',')]
            else:
                # Fall back to space separation
                parts = arg_string.split()

            if len(parts) >= 3:
                ticker = parts[0]
                period = int(parts[1]) if parts[1].isdigit() else 14
                data_period = parts[2]
            elif len(parts) >= 2:
                ticker = parts[0]
                # Check if second part is a number (period) or period string
                if parts[1].isdigit():
                    period = int(parts[1])
                    data_period = "1mo"  # Default data period
                else:
                    period = 14  # Default period
                    data_period = parts[1]  # Use as data period
            else:
                ticker = parts[0]
                period = 14  # Default period
                data_period = "1mo"  # Default data period
        elif len(args) >= 3:
            ticker = args[0]
            period = int(args[1]) if isinstance(args[1], (int, str)) and str(args[1]).isdigit() else 14
            data_period = args[2]
        elif len(args) >= 2:
            ticker = args[0]
            # Check if second arg is a number (period) or period string
            if isinstance(args[1], (int, str)) and str(args[1]).isdigit():
                period = int(args[1])
                data_period = "1mo"  # Default data period
            else:
                period = 14  # Default period
                data_period = args[1]  # Use as data period
        else:
            ticker = kwargs.get('ticker')
            period = kwargs.get('period', 14)
            data_period = kwargs.get('data_period', '1mo')

        if not ticker:
            return "Error: Ticker symbol is required"

        return self._run(ticker, period, data_period)

class CalculateMovingAverageTool(BaseTool):
    name: str = "calculate_moving_average"
    description: str = "Calculate Simple Moving Average (SMA) for stock data. Fetches data first if needed."
    args_schema: type = CalculateMovingAverageInput
    type: str = "function"

    def _run(self, ticker: str, period: int = 20, data_period: str = None) -> str:
        """Calculate moving average."""
        try:
            # Use provided data_period or default to 1mo if not specified
            if data_period is None:
                data_period = "1mo"

            logger.info(f"Calculating moving average for {ticker} with period {period} over {data_period}")

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

                return f"Moving average for {ticker} over {data_period}: {period}-period MA is ${latest_ma:.2f}, current price is ${latest_close:.2f} ({position} MA)"
            else:
                return f"Insufficient data for moving average calculation. Need at least {period} data points."

        except Exception as e:
            logger.error(f"Error calculating moving average: {e}")
            return f"Error calculating moving average: {str(e)}"

    def run(self, *args, **kwargs) -> str:
        """Run the tool with flexible argument handling."""
        # Handle case where arguments might be passed as a single string
        if len(args) == 1 and isinstance(args[0], str):
            # First try to split by pipes (LangChain format), then colons, then commas, then spaces
            arg_string = args[0].strip()

            # Try pipe separation first (LangChain format)
            if '|' in arg_string:
                parts = [part.strip() for part in arg_string.split('|')]
            elif ':' in arg_string:
                parts = [part.strip() for part in arg_string.split(':')]
            elif ',' in arg_string:
                parts = [part.strip() for part in arg_string.split(',')]
            else:
                # Fall back to space separation
                parts = arg_string.split()

            if len(parts) >= 3:
                ticker = parts[0]
                period = int(parts[1]) if parts[1].isdigit() else 20
                data_period = parts[2]
            elif len(parts) >= 2:
                ticker = parts[0]
                # Check if second part is a number (period) or period string
                if parts[1].isdigit():
                    period = int(parts[1])
                    data_period = "1mo"  # Default data period
                else:
                    period = 20  # Default period
                    data_period = parts[1]  # Use as data period
            else:
                ticker = parts[0]
                period = 20  # Default period
                data_period = "1mo"  # Default data period
        elif len(args) >= 3:
            ticker = args[0]
            period = int(args[1]) if isinstance(args[1], (int, str)) and str(args[1]).isdigit() else 20
            data_period = args[2]
        elif len(args) >= 2:
            ticker = args[0]
            # Check if second arg is a number (period) or period string
            if isinstance(args[1], (int, str)) and str(args[1]).isdigit():
                period = int(args[1])
                data_period = "1mo"  # Default data period
            else:
                period = 20  # Default period
                data_period = args[1]  # Use as data period
        else:
            ticker = kwargs.get('ticker')
            period = kwargs.get('period', 20)
            data_period = kwargs.get('data_period', '1mo')

        if not ticker:
            return "Error: Ticker symbol is required"

        return self._run(ticker, period, data_period)

class CalculateBollingerBandsTool(BaseTool):
    name: str = "calculate_bollinger_bands"
    description: str = "Calculate Bollinger Bands for stock data. Fetches data first if needed."
    args_schema: type = CalculateBollingerBandsInput
    type: str = "function"

    def _run(self, ticker: str, period: int = 20, std_dev: float = 2.0, data_period: str = None) -> str:
        """Calculate Bollinger Bands."""
        try:
            # Use provided data_period or default to 1mo if not specified
            if data_period is None:
                data_period = "1mo"

            logger.info(f"Calculating Bollinger Bands for {ticker} with period {period} over {data_period}")

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

                return f"Bollinger Bands for {ticker} over {data_period}: Upper ${latest_upper:.2f}, Middle ${latest_middle:.2f}, Lower ${latest_lower:.2f}. Price is {position}."
            else:
                return f"Insufficient data for Bollinger Bands calculation. Need at least {period} data points."

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return f"Error calculating Bollinger Bands: {str(e)}"

    def run(self, *args, **kwargs) -> str:
        """Run the tool with flexible argument handling."""
        # Handle case where arguments might be passed as a single string
        if len(args) == 1 and isinstance(args[0], str):
            # First try to split by colons (LangChain format), then commas, then spaces
            arg_string = args[0].strip()

            # Try colon separation first (LangChain format)
            if ':' in arg_string:
                parts = [part.strip() for part in arg_string.split(':')]
            elif ',' in arg_string:
                parts = [part.strip() for part in arg_string.split(',')]
            else:
                # Fall back to space separation
                parts = arg_string.split()

            if len(parts) >= 3:
                ticker = parts[0]
                period = int(parts[1]) if parts[1].isdigit() else 20
                data_period = parts[2]
            elif len(parts) >= 2:
                ticker = parts[0]
                # Check if second part is a number (period) or period string
                if parts[1].isdigit():
                    period = int(parts[1])
                    data_period = "1mo"  # Default data period
                else:
                    period = 20  # Default period
                    data_period = parts[1]  # Use as data period
            else:
                ticker = parts[0]
                period = 20  # Default period
                data_period = "1mo"  # Default data period
        elif len(args) >= 3:
            ticker = args[0]
            period = int(args[1]) if isinstance(args[1], (int, str)) and str(args[1]).isdigit() else 20
            data_period = args[2]
        elif len(args) >= 2:
            ticker = args[0]
            # Check if second arg is a number (period) or period string
            if isinstance(args[1], (int, str)) and str(args[1]).isdigit():
                period = int(args[1])
                data_period = "1mo"  # Default data period
            else:
                period = 20  # Default period
                data_period = args[1]  # Use as data period
        else:
            ticker = kwargs.get('ticker')
            period = kwargs.get('period', 20)
            data_period = kwargs.get('data_period', '1mo')

        if not ticker:
            return "Error: Ticker symbol is required"

        std_dev = kwargs.get('std_dev', 2.0)
        return self._run(ticker, period, std_dev, data_period)

class CalculateMACDTool(BaseTool):
    name: str = "calculate_macd"
    description: str = "Calculate MACD (Moving Average Convergence Divergence) for stock data. Fetches data first if needed."
    args_schema: type = CalculateMACDInput
    type: str = "function"

    def _run(self, ticker: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, data_period: str = None) -> str:
        """Calculate MACD."""
        try:
            # Use provided data_period or intelligently choose based on requirements
            if data_period is None:
                # MACD needs at least slow_period (26) + signal_period (9) = 35 data points
                # 1mo (30 days) might be too short, so default to 3mo for safety
                data_period = "3mo"
            else:
                # Validate that the requested period has enough data
                min_required = slow_period + signal_period  # 26 + 9 = 35
                period_days = self._get_period_days(data_period)
                if period_days < min_required:
                    # Suggest a better period
                    suggested_period = self._suggest_period_for_macd(min_required)
                    logger.warning(f"Requested period {data_period} may be too short for MACD. Using {suggested_period} instead.")
                    data_period = suggested_period

            logger.info(f"Calculating MACD for {ticker} over {data_period}")

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

                return f"MACD for {ticker} over {data_period}: MACD {latest_macd:.4f}, Signal {latest_signal:.4f}, Histogram {latest_histogram:.4f}. Signal is {signal}."
            else:
                return f"Insufficient data for MACD calculation. Need at least {slow_period} data points."

        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return f"Error calculating MACD: {str(e)}"

    def _get_period_days(self, period: str) -> int:
        """Get approximate number of days for a period string."""
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        return period_map.get(period, 90)  # Default to 90 days if unknown

    def _suggest_period_for_macd(self, min_required: int) -> str:
        """Suggest an appropriate period for MACD calculation."""
        if min_required <= 30:
            return "1mo"
        elif min_required <= 90:
            return "3mo"
        elif min_required <= 180:
            return "6mo"
        else:
            return "1y"

    def run(self, *args, **kwargs) -> str:
        """Run the tool with flexible argument handling."""
        # Handle case where arguments might be passed as a single string
        if len(args) == 1 and isinstance(args[0], str):
            # First try to split by colons (LangChain format), then commas, then spaces
            arg_string = args[0].strip()

            # Try colon separation first (LangChain format)
            if ':' in arg_string:
                parts = [part.strip() for part in arg_string.split(':')]
            elif ',' in arg_string:
                parts = [part.strip() for part in arg_string.split(',')]
            else:
                # Fall back to space separation
                parts = arg_string.split()

            if len(parts) >= 3:
                ticker = parts[0]
                fast_period = int(parts[1]) if parts[1].isdigit() else 12
                data_period = parts[2]
            elif len(parts) >= 2:
                ticker = parts[0]
                # Check if second part is a number (period) or period string
                if parts[1].isdigit():
                    fast_period = int(parts[1])
                    data_period = "3mo"  # Default data period for MACD
                else:
                    fast_period = 12  # Default period
                    data_period = parts[1]  # Use as data period
            else:
                ticker = parts[0]
                fast_period = 12  # Default period
                data_period = "3mo"  # Default data period for MACD
        elif len(args) >= 3:
            ticker = args[0]
            fast_period = int(args[1]) if isinstance(args[1], (int, str)) and str(args[1]).isdigit() else 12
            data_period = args[2]
        elif len(args) >= 2:
            ticker = args[0]
            # Check if second arg is a number (period) or period string
            if isinstance(args[1], (int, str)) and str(args[1]).isdigit():
                fast_period = int(args[1])
                data_period = "3mo"  # Default data period for MACD
            else:
                fast_period = 12  # Default period
                data_period = args[1]  # Use as data period
        else:
            ticker = kwargs.get('ticker')
            fast_period = kwargs.get('fast_period', 12)
            data_period = kwargs.get('data_period', '3mo')

        if not ticker:
            return "Error: Ticker symbol is required"

        slow_period = kwargs.get('slow_period', 26)
        signal_period = kwargs.get('signal_period', 9)
        return self._run(ticker, fast_period, slow_period, signal_period, data_period)

class CalculateAveragePriceTool(BaseTool):
    name: str = "calculate_average_price"
    description: str = "Calculate the average closing price for all available data points. Fetches data first if needed."
    args_schema: type = CalculateAveragePriceInput
    type: str = "function"

    def _run(self, ticker: str, data_period: str = None) -> str:
        """Calculate average price."""
        try:
            # Use provided data_period or default to 1mo if not specified
            if data_period is None:
                data_period = "1mo"

            logger.info(f"Calculating average price for {ticker} over {data_period}")

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

    def run(self, *args, **kwargs) -> str:
        """Run the tool with flexible argument handling."""
        # Handle case where arguments might be passed as a single string
        if len(args) == 1 and isinstance(args[0], str):
            # First try to split by colons (LangChain format), then commas, then spaces
            arg_string = args[0].strip()

            # Try colon separation first (LangChain format)
            if ':' in arg_string:
                parts = [part.strip() for part in arg_string.split(':')]
            elif ',' in arg_string:
                parts = [part.strip() for part in arg_string.split(',')]
            else:
                # Fall back to space separation
                parts = arg_string.split()

            if len(parts) >= 2:
                ticker = parts[0]
                data_period = parts[1]
            else:
                ticker = parts[0]
                data_period = "1mo"  # Default period
        elif len(args) >= 2:
            ticker = args[0]
            data_period = args[1]
        else:
            ticker = kwargs.get('ticker')
            data_period = kwargs.get('data_period', '1mo')

        if not ticker:
            return "Error: Ticker symbol is required"

        return self._run(ticker, data_period)

class DisplayStockPricesInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol to display prices for")
    data_period: str = Field(description="Data period to fetch and display (e.g., 1d, 5d, 1mo, 3mo)")
    price_type: str = Field(default="close", description="Type of prices to display: 'close', 'open', 'both', or 'all'")

class DisplayStockPricesTool(BaseTool):
    name: str = "display_stock_prices"
    description: str = "Display stock prices in a formatted table. Fetches data and shows daily prices with dates. Can show open, close, or both prices."
    args_schema: type = DisplayStockPricesInput
    type: str = "function"

    def _run(self, ticker: str, data_period: str = None, price_type: str = "close") -> str:
        """Display stock prices in a formatted way."""
        try:
            # Use provided data_period or default to 1mo if not specified
            if data_period is None:
                data_period = "1mo"

            logger.info(f"Displaying stock prices for {ticker} over {data_period}, type: {price_type}")

            # Fetch data first
            df = data_fetcher.fetch_stock_data(ticker, data_period)
            if len(df) == 0:
                return f"No data found for {ticker}"

            # Get the last 30 days of data if we have more
            if len(df) > 30:
                df = df.tail(30)

            # Determine which columns to show based on price_type
            if price_type.lower() in ['open', 'o']:
                show_open = True
                show_close = False
                show_both = False
                price_type_display = "Open"
            elif price_type.lower() in ['close', 'c', 'closing']:
                show_open = False
                show_close = True
                show_both = False
                price_type_display = "Close"
            elif price_type.lower() in ['both', 'b', 'open_close', 'oc']:
                show_open = True
                show_close = True
                show_both = True
                price_type_display = "Open & Close"
            else:
                # Default to close if unknown
                show_open = False
                show_close = True
                show_both = False
                price_type_display = "Close"

            # Format the data for display
            result = f"📊 Stock Prices for {ticker} ({data_period}) - {price_type_display} Prices\n"
            result += f"Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\n"
            result += f"Total Data Points: {len(df)}\n\n"

            # Create header based on what we're showing
            if show_both:
                result += f"{'Date':<12} {'Open':<10} {'Close':<10} {'Change':<10} {'% Change':<10}\n"
                result += "-" * 55 + "\n"
            else:
                result += f"{'Date':<12} {'Price':<10} {'Change':<10} {'% Change':<10}\n"
                result += "-" * 45 + "\n"

            # Calculate changes
            prev_price = None
            for date, row in df.iterrows():
                if show_both:
                    open_price = row['Open']
                    close_price = row['Close']

                    if prev_price is not None:
                        change = close_price - prev_price
                        pct_change = (change / prev_price) * 100
                        change_str = f"${change:+.2f}"
                        pct_str = f"{pct_change:+.2f}%"
                    else:
                        change_str = "N/A"
                        pct_str = "N/A"

                    result += f"{date.strftime('%Y-%m-%d'):<12} ${open_price:<9.2f} ${close_price:<9.2f} {change_str:<10} {pct_str:<10}\n"
                    prev_price = close_price
                else:
                    # Show single price type
                    if show_open:
                        current_price = row['Open']
                        price_label = "Open"
                    else:
                        current_price = row['Close']
                        price_label = "Close"

                    if prev_price is not None:
                        change = current_price - prev_price
                        pct_change = (change / prev_price) * 100
                        change_str = f"${change:+.2f}"
                        pct_str = f"{pct_change:+.2f}%"
                    else:
                        change_str = "N/A"
                        pct_str = "N/A"

                    result += f"{date.strftime('%Y-%m-%d'):<12} ${current_price:<9.2f} {change_str:<10} {pct_str:<10}\n"
                    prev_price = current_price

            # Add summary
            if show_both:
                latest_open = df['Open'].iloc[-1]
                latest_close = df['Close'].iloc[-1]
                first_open = df['Open'].iloc[0]
                first_close = df['Close'].iloc[0]

                result += "\n" + "-" * 55 + "\n"
                result += f"Summary: {ticker} started at Open:${first_open:.2f}/Close:${first_close:.2f} and ended at Open:${latest_open:.2f}/Close:${latest_close:.2f}\n"

                open_change = latest_open - first_open
                close_change = latest_close - first_close
                open_pct = (open_change / first_open) * 100
                close_pct = (close_change / first_close) * 100

                result += f"Open Change: ${open_change:+.2f} ({open_pct:+.2f}%)\n"
                result += f"Close Change: ${close_change:+.2f} ({close_pct:+.2f}%)\n"
            else:
                if show_open:
                    latest_price = df['Open'].iloc[-1]
                    first_price = df['Open'].iloc[0]
                    price_label = "Open"
                else:
                    latest_price = df['Close'].iloc[-1]
                    first_price = df['Close'].iloc[0]
                    price_label = "Close"

                total_change = latest_price - first_price
                total_pct_change = (total_change / first_price) * 100

                result += "\n" + "-" * 45 + "\n"
                result += f"Summary: {ticker} started at ${first_price:.2f} and ended at ${latest_price:.2f}\n"
                result += f"Total Change: ${total_change:+.2f} ({total_pct_change:+.2f}%)\n"

            return result

        except Exception as e:
            logger.error(f"Error displaying stock prices: {e}")
            return f"Error displaying stock prices: {str(e)}"

    def run(self, *args, **kwargs) -> str:
        """Run the tool with flexible argument handling."""
        # Handle case where arguments might be passed as a single string
        if len(args) == 1 and isinstance(args[0], str):
            # First try to split by colons (LangChain format), then commas, then spaces
            arg_string = args[0].strip()

            # Try colon separation first (LangChain format)
            if ':' in arg_string:
                parts = [part.strip() for part in arg_string.split(':')]
            elif ',' in arg_string:
                parts = [part.strip() for part in arg_string.split(',')]
            else:
                # Fall back to space separation
                parts = arg_string.split()

            if len(parts) >= 3:
                ticker = parts[0]
                data_period = parts[1]
                price_type = parts[2]
            elif len(parts) >= 2:
                ticker = parts[0]
                data_period = parts[1]
                price_type = "close"  # Default to close
            else:
                ticker = parts[0]
                data_period = "1mo"  # Default period
                price_type = "close"  # Default to close
        elif len(args) >= 3:
            ticker = args[0]
            data_period = args[1]
            price_type = args[2]
        elif len(args) >= 2:
            ticker = args[0]
            data_period = args[1]
            price_type = "close"
        else:
            ticker = kwargs.get('ticker')
            data_period = kwargs.get('data_period', '1mo')
            price_type = kwargs.get('price_type', 'close')

        if not ticker:
            return "Error: Ticker symbol is required"

        return self._run(ticker, data_period, price_type)

    def get_output_signature(self) -> str:
        """Return a signature that can be used to identify this tool's output."""
        return "�� Stock Prices for"

def get_all_tools():
    """Get all available LangChain tools."""
    return [
        FetchStockDataTool(),
        CalculateRSITool(),
        CalculateMovingAverageTool(),
        CalculateBollingerBandsTool(),
        CalculateMACDTool(),
        CalculateAveragePriceTool(),
        DisplayStockPricesTool()
    ]
