"""
Demo script for the modular Stock Analysis AI system

Shows how to use the system programmatically with custom prompts.
"""
import logging
from stock_agent import StockAnalysisAgent
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_custom_prompts():
    """Demonstrate the system with various custom prompts."""
    print("🚀 Stock Analysis AI - Modular System Demo")
    print("=" * 60)

    try:
        # Initialize the agent
        print("Initializing Stock Analysis Agent...")
        agent = StockAnalysisAgent()
        print("✅ Agent initialized successfully!")

        # Show available capabilities
        print("\n📚 Available Tools:")
        tools = agent.get_available_tools()
        for tool, desc in tools.items():
            print(f"  - {tool}: {desc}")

        print("\n⏰ Available Time Periods:")
        periods = agent.get_available_periods()
        for period, desc in periods.items():
            print(f"  - {period}: {desc}")

        # Example custom prompts
        custom_prompts = [
            "Get me Tesla stock for last month and calculate RSI",
            "Fetch Apple stock data for 3 months and calculate moving average with period 20",
            "Get Microsoft stock for last year and calculate Bollinger Bands",
            "Show me Google stock data for 6 months and calculate MACD",
            "Analyze Amazon stock for the last 2 years with RSI and moving averages"
        ]

        print(f"\n💡 Testing {len(custom_prompts)} custom prompts...")
        print("=" * 60)

        for i, prompt in enumerate(custom_prompts, 1):
            print(f"\n🔍 Test {i}: {prompt}")
            print("-" * 40)

            try:
                result = agent.analyze(prompt)

                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print("✅ Analysis completed successfully!")
                    print(f"   Actions executed: {len(result['call_log'])}")

                    # Show some key results
                    for key, value in result['results'].items():
                        if isinstance(value, pd.DataFrame):
                            print(f"   📈 {key}: {len(value)} data points")
                        elif isinstance(value, list):
                            print(f"   📊 {key}: {len(value)} calculated values")
                        elif isinstance(value, dict):
                            print(f"   📊 {key}: {type(value).__name__} with {len(value)} components")

            except Exception as e:
                print(f"❌ Error processing prompt: {str(e)}")
                logger.error(f"Error processing prompt '{prompt}': {str(e)}")

            print("-" * 40)

        print("\n🎉 Demo completed!")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Demo error: {str(e)}")

def demo_specific_analysis():
    """Demonstrate a specific analysis workflow."""
    print("\n🔬 Specific Analysis Demo")
    print("=" * 40)

    try:
        agent = StockAnalysisAgent()

        # Custom prompt for specific analysis
        prompt = "Get me NVIDIA stock for last 6 months and calculate RSI with period 14"
        print(f"Analyzing: {prompt}")

        result = agent.analyze(prompt)

        if 'error' not in result:
            print("✅ Analysis successful!")
            print(f"Results: {list(result['results'].keys())}")

            # Show detailed results
            for key, value in result['results'].items():
                if isinstance(value, pd.DataFrame):
                    print(f"\n📈 Stock Data ({key}):")
                    print(f"   Rows: {len(value)}")
                    print(f"   Date range: {value.index[0].strftime('%Y-%m-%d')} to {value.index[-1].strftime('%Y-%m-%d')}")
                    print(f"   Latest close: ${value['Close'].iloc[-1]:.2f}")
                elif isinstance(value, list):
                    print(f"\n📊 RSI Values ({key}):")
                    print(f"   Count: {len(value)}")
                    print(f"   Latest RSI: {value[-1]:.2f}")
                    print(f"   First 5 values: {value[:5]}")
        else:
            print(f"❌ Analysis failed: {result['error']}")

    except Exception as e:
        print(f"❌ Specific analysis demo failed: {str(e)}")

if __name__ == "__main__":
    # Run the main demo
    demo_custom_prompts()

    # Run specific analysis demo
    demo_specific_analysis()

    print("\n🏁 All demos completed!")
