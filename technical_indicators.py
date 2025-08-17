"""
Technical Indicators Module

Handles calculation of various technical analysis indicators like RSI, Moving Averages, and Bollinger Bands.
"""
import logging
import pandas as pd
from typing import Union, List, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculates various technical analysis indicators."""

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index (RSI) for stock data.

        Args:
            data (pd.DataFrame): Stock data DataFrame with 'Close' column
            period (int): RSI calculation period (default: 14)

        Returns:
            List of RSI values

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")

            if 'Close' not in data.columns:
                raise ValueError("Close price column not found in data")

            if len(data) < period:
                logger.warning(f"Insufficient data for RSI calculation: {len(data)} points available, {period} required")
                logger.warning(f"Consider fetching data with a longer period (e.g., '1mo' instead of '5d')")
                return []

            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Remove NaN values and convert to list
            rsi_values = rsi.dropna().tolist()

            logger.info(f"RSI calculated successfully for {len(rsi_values)} periods")
            return rsi_values

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise ValueError(f"Error calculating RSI: {str(e)}")

    @staticmethod
    def calculate_moving_average(data: pd.DataFrame, period: int = 20) -> List[float]:
        """
        Calculate Simple Moving Average (SMA) for stock data.

        Args:
            data (pd.DataFrame): Stock data DataFrame with 'Close' column
            period (int): Moving average period (default: 20)

        Returns:
            List of moving average values

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")

            if 'Close' not in data.columns:
                raise ValueError("Close price column not found in data")

            if len(data) < period:
                logger.warning(f"Insufficient data for moving average calculation: {len(data)} points available, {period} required")
                logger.warning(f"Consider fetching data with a longer period (e.g., '1mo' instead of '5d')")
                return []

            # Calculate Simple Moving Average
            sma = data['Close'].rolling(window=period).mean()

            # Remove NaN values and convert to list
            sma_values = sma.dropna().tolist()

            logger.info(f"Moving average calculated successfully for {len(sma_values)} periods")
            return sma_values

        except Exception as e:
            logger.error(f"Error calculating moving average: {str(e)}")
            raise ValueError(f"Error calculating moving average: {str(e)}")

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, List[float]]:
        """
        Calculate Bollinger Bands for stock data.

        Args:
            data (pd.DataFrame): Stock data DataFrame with 'Close' column
            period (int): Moving average period (default: 20)
            std_dev (float): Standard deviation multiplier (default: 2.0)

        Returns:
            Dictionary with upper, middle, and lower bands

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")

            if 'Close' not in data.columns:
                raise ValueError("Close price column not found in data")

            # Calculate Bollinger Bands
            sma = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            # Remove NaN values and convert to lists
            upper_values = upper_band.dropna().tolist()
            middle_values = sma.dropna().tolist()
            lower_values = lower_band.dropna().tolist()

            result = {
                "upper_band": upper_values,
                "middle_band": middle_values,
                "lower_band": lower_values
            }

            logger.info(f"Bollinger Bands calculated successfully for {len(upper_values)} periods")
            return result

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise ValueError(f"Error calculating Bollinger Bands: {str(e)}")

    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence) for stock data.

        Args:
            data (pd.DataFrame): Stock data DataFrame with 'Close' column
            fast_period (int): Fast EMA period (default: 12)
            slow_period (int): Slow EMA period (default: 26)
            signal_period (int): Signal line period (default: 9)

        Returns:
            Dictionary with MACD line, signal line, and histogram

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")

            if 'Close' not in data.columns:
                raise ValueError("Close price column not found in data")

            # Calculate EMAs
            ema_fast = data['Close'].ewm(span=fast_period).mean()
            ema_slow = data['Close'].ewm(span=slow_period).mean()

            # Calculate MACD line
            macd_line = ema_fast - ema_slow

            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period).mean()

            # Calculate histogram
            histogram = macd_line - signal_line

            # Remove NaN values and convert to lists
            macd_values = macd_line.dropna().tolist()
            signal_values = signal_line.dropna().tolist()
            histogram_values = histogram.dropna().tolist()

            result = {
                "macd_line": macd_values,
                "signal_line": signal_values,
                "histogram": histogram_values
            }

            logger.info(f"MACD calculated successfully for {len(macd_values)} periods")
            return result

        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise ValueError(f"Error calculating MACD: {str(e)}")

    @staticmethod
    def calculate_average_price(data: pd.DataFrame) -> float:
        """
        Calculate the average closing price for all available data points.

        Args:
            data (pd.DataFrame): Stock data DataFrame with 'Close' column

        Returns:
            float: Average closing price

        Raises:
            ValueError: If data is invalid or missing required columns
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")

            if 'Close' not in data.columns:
                raise ValueError("Close price column not found in data")

            if len(data) == 0:
                raise ValueError("DataFrame is empty")

            # Calculate simple average of all closing prices
            average_price = data['Close'].mean()

            logger.info(f"Average price calculated successfully: ${average_price:.2f} for {len(data)} data points")
            return average_price

        except Exception as e:
            logger.error(f"Error calculating average price: {str(e)}")
            raise ValueError(f"Error calculating average price: {str(e)}")

    @staticmethod
    def get_available_indicators() -> Dict[str, str]:
        """Get available technical indicators and their descriptions."""
        return {
            "rsi": "Relative Strength Index - measures momentum and overbought/oversold conditions",
            "moving_average": "Simple Moving Average - shows trend direction and support/resistance",
            "bollinger_bands": "Bollinger Bands - shows volatility and potential reversal points",
            "macd": "MACD - shows trend changes and momentum shifts"
        }
