"""
AI-Powered Stock Analysis Agent

This module provides an intelligent stock analysis system that uses OpenAI's API
to parse natural language prompts and automatically execute technical analysis tools.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Union
from datetime import datetime
import warnings

# Import our modular components
from config import OPENAI_API_KEY, POLYGON_API_KEY
from data_fetcher import StockDataFetcher
from technical_indicators import TechnicalIndicators
from prompt_parser import PromptParser

# Suppress SSL warnings
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalysisAgent:
    """
    AI-powered stock analysis agent that uses OpenAI to parse natural language
    requests and execute technical analysis tools.
    """

    def __init__(self):
        """Initialize the agent with modular components."""
        # Validate API keys
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if not POLYGON_API_KEY:
            logger.warning("POLYGON_API_KEY not found - stock data fetching will fail")

        # Initialize modular components
        self.data_fetcher = StockDataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.prompt_parser = PromptParser()

        # Map tool names to actual functions
        self.tools = {
            "fetch_stock_data": self.data_fetcher.fetch_stock_data,
            "calculate_rsi": self.technical_indicators.calculate_rsi,
            "calculate_moving_average": self.technical_indicators.calculate_moving_average,
            "calculate_bollinger_bands": self.technical_indicators.calculate_bollinger_bands,
            "calculate_macd": self.technical_indicators.calculate_macd,
            "calculate_average_price": self.technical_indicators.calculate_average_price,
        }

        # Store execution context
        self.execution_context = {}

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Main method to analyze a natural language prompt and execute the required tools.

        Args:
            prompt (str): Natural language description of what to analyze

        Returns:
            Dict containing results and execution log
        """
        try:
            logger.info(f"Processing prompt: {prompt}")

            # Get execution plan from OpenAI
            plan = self.prompt_parser.parse_prompt(prompt)
            logger.info(f"Execution plan: {plan}")

            # Validate the plan
            if not self.prompt_parser.validate_plan(plan):
                raise ValueError("Invalid execution plan generated")

            # Execute the plan
            results, call_log = self._execute_plan(plan)

            return {
                "results": results,
                "call_log": call_log,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {
                "error": str(e),
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }

    def _execute_plan(self, plan: List[Dict[str, Any]]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute the action plan and return results with execution log.

        Args:
            plan (List[Dict]): List of actions to execute

        Returns:
            Tuple of (results, call_log)
        """
        results = {}
        call_log = []

        for i, action in enumerate(plan):
            tool_name = action.get('tool')
            args = action.get('args', {})
            description = action.get('description', 'No description')

            logger.info(f"Executing action {i+1}: {tool_name} - {description}")

            # Check if tool exists
            if tool_name not in self.tools:
                log_entry = {
                    "tool": tool_name,
                    "args": args,
                    "description": description,
                    "status": "not found",
                    "result": "Tool not available",
                    "timestamp": datetime.now().isoformat()
                }
                call_log.append(log_entry)
                results[tool_name] = "Tool not available"
                continue

            # Execute the tool
            try:
                # Replace references to previous results
                processed_args = self._process_args(args, results)

                tool_result = self.tools[tool_name](**processed_args)

                # Store result for future reference
                results[f"result_of_{tool_name}"] = tool_result

                log_entry = {
                    "tool": tool_name,
                    "args": processed_args,
                    "description": description,
                    "status": "success",
                    "result": "Tool executed successfully",
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Error executing {tool_name}: {str(e)}")
                results[tool_name] = f"Error: {str(e)}"

                log_entry = {
                    "tool": tool_name,
                    "args": args,
                    "description": description,
                    "status": "error",
                    "result": f"Error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }

            call_log.append(log_entry)

        return results, call_log

    def _process_args(self, args: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process arguments, replacing references to previous results.

        Args:
            args (Dict): Original arguments
            results (Dict): Results from previous tool executions

        Returns:
            Dict: Processed arguments
        """
        processed_args = {}

        for key, value in args.items():
            if isinstance(value, str) and value.startswith('result_of_'):
                # Replace reference with actual result
                if value in results:
                    processed_args[key] = results[value]
                else:
                    processed_args[key] = value
            elif isinstance(value, str) and value in results:
                # Handle direct tool name references
                processed_args[key] = results[value]
            else:
                processed_args[key] = value

        return processed_args

    def get_available_tools(self) -> Dict[str, str]:
        """Get available tools and their descriptions."""
        return self.prompt_parser.get_available_tools()

    def get_example_prompts(self) -> List[str]:
        """Get example prompts to help users understand the system."""
        return self.prompt_parser.get_example_prompts()

    def get_available_periods(self) -> Dict[str, str]:
        """Get available time periods for data fetching."""
        return self.data_fetcher.get_available_periods()

    def get_available_indicators(self) -> Dict[str, str]:
        """Get available technical indicators and their descriptions."""
        return self.technical_indicators.get_available_indicators()
