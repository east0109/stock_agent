"""
Stock Data Fetcher Module

Handles fetching stock data from Polygon.io API and data validation.
"""
import logging
import requests
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta
from config import POLYGON_API_KEY, POLYGON_BASE_URL

# Set up logging
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Handles fetching stock data from various sources."""

    def __init__(self):
        """Initialize the data fetcher."""
        if not POLYGON_API_KEY:
            logger.warning("POLYGON_API_KEY not found - stock data fetching will fail")

    def fetch_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch stock data for a given ticker symbol using Polygon.io.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
            period (str): Time period for data (default: '1y')
                         Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'

        Returns:
            pd.DataFrame: DataFrame with OHLCV (Open, High, Low, Close, Volume) data

        Raises:
            ValueError: If ticker is invalid or data cannot be fetched
        """
        try:
            # Validate ticker input
            if not ticker or not isinstance(ticker, str):
                raise ValueError("Ticker must be a non-empty string")

            ticker = ticker.upper().strip()
            logger.info(f"Fetching data for ticker: {ticker}")

            # Calculate date range based on period
            end_date = datetime.now()

            days_map = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
                '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
            }

            days_to_fetch = days_map.get(period, 365)  # Default to 1 year
            start_date = end_date - timedelta(days=days_to_fetch)

            # Format dates for Polygon.io API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Make API request to Polygon.io
            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
            params = {
                'apiKey': POLYGON_API_KEY,
                'adjusted': 'true',  # Get adjusted prices
                'sort': 'asc'        # Sort by date ascending
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if data.get('status') not in ['OK', 'DELAYED']:
                error_msg = data.get('error', 'Unknown error')
                raise ValueError(f"Polygon API Error: {error_msg}")

            # Extract results
            results = data.get('results', [])
            if not results:
                # If 1-day request fails, try with 5 days to get recent data
                if period == '1d':
                    logger.info("1-day request returned no data, trying with 5 days to get recent data")
                    return self.fetch_stock_data(ticker, '5d')
                else:
                    raise ValueError(f"No data found for ticker: {ticker} over period: {period}")

            # Convert to DataFrame
            df_data = []
            for item in results:
                df_data.append({
                    'Date': pd.to_datetime(item['t'], unit='ms'),
                    'Open': item['o'],
                    'High': item['h'],
                    'Low': item['l'],
                    'Close': item['c'],
                    'Volume': item['v']
                })

            # Create DataFrame and set index
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)

            logger.info(f"Successfully fetched {len(df)} data points for {ticker}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching data for {ticker}: {str(e)}")
            raise ValueError(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise ValueError(f"Error fetching data: {str(e)}")

    def get_available_periods(self) -> Dict[str, str]:
        """Get available time periods for data fetching."""
        return {
            '1d': '1 day',
            '5d': '5 days',
            '1mo': '1 month',
            '3mo': '3 months',
            '6mo': '6 months',
            '1y': '1 year',
            '2y': '2 years',
            '5y': '5 years',
            '10y': '10 years',
            'ytd': 'Year to date',
            'max': 'Maximum available'
        }
