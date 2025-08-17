"""
Prompt Parser Module

Handles parsing user prompts using OpenAI and creating execution plans.
"""
import json
import logging
import openai
from typing import Dict, List, Any
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS

# Set up logging
logger = logging.getLogger(__name__)

class PromptParser:
    """Parses natural language prompts and creates execution plans using OpenAI."""

    def __init__(self):
        """Initialize the prompt parser."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")

        # Initialize OpenAI client for the latest API
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Available tools and their descriptions with exact parameter names
        self.tool_descriptions = {
            "fetch_stock_data": "Fetches stock data for a given ticker and time period. Parameters: ticker (str), period (str: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max). Note: Use '5d' for 'latest' requests to ensure data availability.",
            "calculate_rsi": "Calculates Relative Strength Index for stock data. Parameters: data (DataFrame), period (int, default: 14)",
            "calculate_moving_average": "Calculates Simple Moving Average for stock data. Parameters: data (DataFrame), period (int, default: 20)",
            "calculate_bollinger_bands": "Calculates Bollinger Bands for stock data. Parameters: data (DataFrame), period (int, default: 20), std_dev (float, default: 2.0)",
            "calculate_macd": "Calculates MACD (Moving Average Convergence Divergence) for stock data. Parameters: data (DataFrame), fast_period (int, default: 12), slow_period (int, default: 26), signal_period (int, default: 9)",
            "calculate_average_price": "Calculates the average closing price for all available data points. Parameters: data (DataFrame)"
        }

    def parse_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Parse a natural language prompt and return a structured execution plan.

        Args:
            prompt (str): Natural language prompt from user

        Returns:
            List of actions to execute

        Raises:
            ValueError: If parsing fails or response is invalid
        """
        try:
            logger.info(f"Parsing prompt: {prompt}")

            # Create system message with tool descriptions and exact parameter requirements
            system_message = f"""You are a stock analysis assistant. Parse user requests and return a JSON plan.

Available tools:
{json.dumps(self.tool_descriptions, indent=2)}

Rules:
1. Use exact parameter names shown above
2. For "latest" requests, use period "5d"
3. For technical indicators: RSI/MA/BB need "1mo", MACD needs "3mo"
4. Use "result_of_fetch_stock_data" for references
5. For simple averages, use calculate_average_price

Return JSON array with actions:
[
  {{
    "tool": "fetch_stock_data",
    "args": {{"ticker": "TICKER", "period": "PERIOD"}},
    "description": "Description"
  }}
]

User request: {prompt}"""

            # Use the latest OpenAI API with client - keep it simple like minimal_test.py
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_message}
                ],
                max_completion_tokens=5000
            )

            # Extract the response content
            content = response.choices[0].message.content.strip()

            # Debug: Log the raw response content
            logger.info(f"Raw OpenAI response: {repr(content)}")
            logger.info(f"Response length: {len(content)}")
            logger.info(f"Response type: {type(content)}")

            # Try to parse JSON from the response
            try:
                # Remove any markdown formatting
                if content.startswith('```json'):
                    content = content.split('```json')[1]
                if content.endswith('```'):
                    content = content.rsplit('```', 1)[0]

                plan = json.loads(content)
                if not isinstance(plan, list):
                    raise ValueError("Plan must be a list")

                logger.info(f"Successfully parsed prompt into {len(plan)} actions")
                return plan

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {content}")
                logger.error(f"JSON error details: {str(e)}")
                raise ValueError(f"Invalid response format from OpenAI: {str(e)}")

        except Exception as e:
            logger.error(f"Error parsing prompt: {str(e)}")
            raise

    def validate_plan(self, plan: List[Dict[str, Any]]) -> bool:
        """
        Validate that the execution plan is properly formatted.

        Args:
            plan (List[Dict]): Execution plan to validate

        Returns:
            bool: True if plan is valid, False otherwise
        """
        try:
            if not isinstance(plan, list):
                return False

            for action in plan:
                if not isinstance(action, dict):
                    return False

                required_fields = ['tool', 'args', 'description']
                for field in required_fields:
                    if field not in action:
                        return False

                if action['tool'] not in self.tool_descriptions:
                    return False

            return True

        except Exception:
            return False

    def get_available_tools(self) -> Dict[str, str]:
        """Get available tools and their descriptions."""
        return self.tool_descriptions.copy()

    def get_example_prompts(self) -> List[str]:
        """Get example prompts to help users understand the system."""
        return [
            "Get me Tesla stock for last month and calculate RSI",
            "Fetch Apple stock data for 3 months and calculate moving average with period 20",
            "Get Microsoft stock for last year and calculate Bollinger Bands",
            "Show me Google stock data for 6 months and calculate MACD",
            "Analyze Amazon stock for the last 2 years with RSI and moving averages"
        ]
