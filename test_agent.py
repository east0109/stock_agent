"""
Test script for the modular Stock Analysis AI system

Tests the basic functionality of each module.
"""
import sys
import logging
from unittest.mock import Mock, patch
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.ERROR)  # Reduce noise during tests
logger = logging.getLogger(__name__)

def test_config_module():
    """Test the configuration module."""
    print("🧪 Testing config module...")
    try:
        from config import OPENAI_API_KEY, POLYGON_API_KEY, OPENAI_MODEL
        print("  ✅ Config module imported successfully")
        print(f"  📝 OpenAI Model: {OPENAI_MODEL}")
        return True
    except Exception as e:
        print(f"  ❌ Config module failed: {str(e)}")
        return False

def test_data_fetcher_module():
    """Test the data fetcher module."""
    print("🧪 Testing data fetcher module...")
    try:
        from data_fetcher import StockDataFetcher

        # Test initialization
        fetcher = StockDataFetcher()
        print("  ✅ Data fetcher initialized successfully")

        # Test available periods
        periods = fetcher.get_available_periods()
        print(f"  📅 Available periods: {len(periods)}")

        return True
    except Exception as e:
        print(f"  ❌ Data fetcher module failed: {str(e)}")
        return False

def test_technical_indicators_module():
    """Test the technical indicators module."""
    print("🧪 Testing technical indicators module...")
    try:
        from technical_indicators import TechnicalIndicators

        # Test initialization
        indicators = TechnicalIndicators()
        print("  ✅ Technical indicators initialized successfully")

        # Test available indicators
        available_indicators = indicators.get_available_indicators()
        print(f"  📊 Available indicators: {len(available_indicators)}")

        # Test with mock data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        })

        # Test RSI calculation
        rsi_values = indicators.calculate_rsi(mock_data, period=5)
        print(f"  📈 RSI calculation test: {len(rsi_values)} values")

        return True
    except Exception as e:
        print(f"  ❌ Technical indicators module failed: {str(e)}")
        return False

def test_prompt_parser_module():
    """Test the prompt parser module (with mocked OpenAI)."""
    print("🧪 Testing prompt parser module...")
    try:
        from prompt_parser import PromptParser

        # Mock OpenAI API key for testing
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            parser = PromptParser()
            print("  ✅ Prompt parser initialized successfully")

            # Test available tools
            tools = parser.get_available_tools()
            print(f"  🔧 Available tools: {len(tools)}")

            # Test example prompts
            examples = parser.get_example_prompts()
            print(f"  💡 Example prompts: {len(examples)}")

        return True
    except Exception as e:
        print(f"  ❌ Prompt parser module failed: {str(e)}")
        return False

def test_stock_agent_module():
    """Test the main stock agent module."""
    print("🧪 Testing stock agent module...")
    try:
        from stock_agent import StockAnalysisAgent

        # Mock API keys for testing
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key', 'POLYGON_API_KEY': 'test_key'}):
            agent = StockAnalysisAgent()
            print("  ✅ Stock agent initialized successfully")

            # Test available methods
            tools = agent.get_available_tools()
            periods = agent.get_available_periods()
            indicators = agent.get_available_indicators()

            print(f"  🔧 Tools: {len(tools)}")
            print(f"  📅 Periods: {len(periods)}")
            print(f"  📊 Indicators: {len(indicators)}")

        return True
    except Exception as e:
        print(f"  ❌ Stock agent module failed: {str(e)}")
        return False

def test_modular_imports():
    """Test that all modules can be imported together."""
    print("🧪 Testing modular imports...")
    try:
        # Test importing all modules
        import config
        import data_fetcher
        import technical_indicators
        import prompt_parser
        import stock_agent

        print("  ✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ Modular imports failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Stock Analysis AI - Modular System Tests")
    print("=" * 50)

    tests = [
        test_config_module,
        test_data_fetcher_module,
        test_technical_indicators_module,
        test_prompt_parser_module,
        test_stock_agent_module,
        test_modular_imports
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test crashed: {str(e)}")
            print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The modular system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
