"""
Main Entry Point for Stock Analysis AI

Provides a user interface for custom prompts and demonstrates the modular system.
"""
import sys
import logging
from stock_agent import StockAnalysisAgent
import pandas as pd
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_banner():
    """Print a welcome banner."""
    print("=" * 80)
    print("🤖 AI-Powered Stock Analysis System")
    print("=" * 80)
    print("This system uses OpenAI to understand your natural language requests")
    print("and automatically executes the appropriate technical analysis tools.")
    print("=" * 80)

def print_help():
    """Print help information."""
    print("\n📚 Available Tools:")
    print("- fetch_stock_data: Get stock data for a ticker and time period")
    print("- calculate_rsi: Calculate Relative Strength Index")
    print("- calculate_moving_average: Calculate Simple Moving Average")
    print("- calculate_bollinger_bands: Calculate Bollinger Bands")
    print("- calculate_macd: Calculate MACD indicator")

    print("\n⏰ Available Time Periods:")
    print("- 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")

    print("\n💡 Example Prompts:")
    print("- 'Get me Tesla stock for last month and calculate RSI'")
    print("- 'Fetch Apple stock data for 3 months and calculate moving average'")
    print("- 'Get Microsoft stock for last year and calculate Bollinger Bands'")
    print("- 'Show me Google stock data for 6 months and calculate MACD'")

    print("\n🔧 Commands:")
    print("- 'help': Show this help information")
    print("- 'tools': Show available tools")
    print("- 'periods': Show available time periods")
    print("- 'examples': Show example prompts")
    print("- 'quit' or 'exit': Exit the program")

def print_results(result: Dict[str, Any]) -> None:
    """Print analysis results in a formatted way."""
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return

    print("\n📊 Analysis Results:")
    print("-" * 40)

    for key, value in result['results'].items():
        if isinstance(value, pd.DataFrame):
            print(f"  📈 {key}: DataFrame with {len(value)} rows")
            if len(value) > 0:
                print(f"     Latest close: ${value['Close'].iloc[-1]:.2f}")
                print(f"     Date range: {value.index[0].strftime('%Y-%m-%d')} to {value.index[-1].strftime('%Y-%m-%d')}")
                print(f"     Data preview:")
                # Show first 5 and last 5 rows for better readability
                if len(value) <= 10:
                    print(value.to_string())
                else:
                    print("     First 5 rows:")
                    print(value.head().to_string())
                    print("     Last 5 rows:")
                    print(value.tail().to_string())
        elif isinstance(value, list) and len(value) > 5:
            print(f"  📊 {key}: List with {len(value)} values")
            print(f"     Latest values: {value[-5:]}")
        elif isinstance(value, dict):
            print(f"  📊 {key}: {type(value).__name__}")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list) and len(sub_value) > 0:
                    if len(sub_value) <= 10:
                        # For small lists, show all values
                        print(f"     {sub_key}: {[f'{x:.2f}' for x in sub_value]}")
                    else:
                        # For larger lists, show first 5 and last 5 values
                        first_values = [f'{x:.2f}' for x in sub_value[:5]]
                        last_values = [f'{x:.2f}' for x in sub_value[-5:]]
                        print(f"     {sub_key}: {len(sub_value)} values")
                        print(f"       First 5: {first_values}")
                        print(f"       Last 5: {last_values}")
                        print(f"       Latest: {sub_value[-1]:.2f}")
                else:
                    print(f"     {sub_key}: {sub_value}")
        else:
            print(f"  📊 {key}: {value}")

    print(f"\n📝 Execution Log ({len(result['call_log'])} actions):")
    print("-" * 40)
    for i, log_entry in enumerate(result['call_log']):
        status_icon = "✅" if log_entry['status'] == 'success' else "❌"
        print(f"  {i+1}. {status_icon} {log_entry['tool']}: {log_entry['status']} - {log_entry['result']}")

def main():
    """Main interactive loop."""
    print_banner()

    try:
        # Initialize the agent
        print("🚀 Initializing Stock Analysis Agent...")
        agent = StockAnalysisAgent()
        print("✅ Agent initialized successfully!")

        print_help()

        # Main interaction loop
        while True:
            try:
                print("\n" + "=" * 80)
                prompt = input("💬 Enter your stock analysis request (or 'help' for options): ").strip()

                if not prompt:
                    continue

                # Handle special commands
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Happy trading!")
                    break
                elif prompt.lower() == 'help':
                    print_help()
                    continue
                elif prompt.lower() == 'tools':
                    tools = agent.get_available_tools()
                    print("\n🔧 Available Tools:")
                    for tool, desc in tools.items():
                        print(f"  - {tool}: {desc}")
                    continue
                elif prompt.lower() == 'periods':
                    periods = agent.get_available_periods()
                    print("\n⏰ Available Time Periods:")
                    for period, desc in periods.items():
                        print(f"  - {period}: {desc}")
                    continue
                elif prompt.lower() == 'examples':
                    examples = agent.get_example_prompts()
                    print("\n💡 Example Prompts:")
                    for i, example in enumerate(examples, 1):
                        print(f"  {i}. {example}")
                    continue

                # Process the analysis request
                print(f"\n🔄 Processing: {prompt}")
                result = agent.analyze(prompt)

                # Display results
                print_results(result)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy trading!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
                logger.error(f"Unexpected error in main loop: {str(e)}")
                continue

    except Exception as e:
        print(f"❌ Failed to initialize Stock Analysis Agent: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
