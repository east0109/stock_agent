#!/usr/bin/env python3
"""
LangChain-Powered Stock Analysis System

This is the main entry point for the LangChain-based stock analysis system.
It provides an interactive interface for users to request stock analysis
using natural language, which the LangChain agent will execute using tools.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_agent import LangChainStockAgent
from config import OPENAI_API_KEY

# Global variable to control tool execution details display
show_tool_details = True

# Global variable to control AI response display
show_ai_response = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_banner():
    """Print the system banner."""
    print("=" * 80)
    print("🤖 LangChain-Powered Stock Analysis System")
    print("=" * 80)
    print("This system uses LangChain and OpenAI GPT-5 to understand your requests")
    print("and automatically execute the appropriate technical analysis tools.")
    print("=" * 80)

def print_help():
    """Print help information."""
    print("\n📚 Available Tools:")
    print("- fetch_stock_data: Get stock data for any ticker and time period")
    print("- calculate_rsi: Calculate Relative Strength Index")
    print("- calculate_moving_average: Calculate Simple Moving Average")
    print("- calculate_bollinger_bands: Calculate Bollinger Bands")
    print("- calculate_macd: Calculate MACD indicator")
    print("- calculate_average_price: Calculate average closing price")

    print("\n⏰ Supported Time Periods:")
    print("- 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")

    print("\n💡 Example Prompts:")
    print("- 'Get me Tesla stock for last month and calculate RSI'")
    print("- 'Fetch Apple stock data for 3 months and calculate moving average'")
    print("- 'Show me NVIDIA stock for last year and calculate MACD'")
    print("- 'Give me this week's stock price for Tesla and its average'")

    print("\n🔧 Commands:")
    print("- 'help': Show this help information")
    print("- 'tools': Show available tools")
    print("- 'history': Show conversation history")
    print("- 'clear': Clear conversation memory")
    print("- 'toggle_details': Toggle tool execution details on/off")
    print("- 'toggle_ai_response': Toggle AI response display on/off")
    print("- 'quit' or 'exit': Exit the program")
    print("\n💡 Tip: Use 'toggle_details' to reduce output duplication when tools return formatted data")
    print("💡 Tip: Use 'toggle_ai_response' to hide AI responses that just repeat tool output")
    print("=" * 80)

def print_results(result: dict) -> None:
    """Print analysis results in a formatted way."""
    if not result.get('success', False):
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return

    print("\n📊 Analysis Results:")
    print("-" * 40)

    # Get the main output and intermediate steps
    main_output = result.get('output', 'No output')
    intermediate_steps = result.get('intermediate_steps', [])

    # Check if the main output is just duplicating tool output
    is_duplicate_output = False
    if intermediate_steps:
        for step in intermediate_steps:
            tool_output = str(step[1])
            # If the main output is essentially the same as tool output, mark as duplicate
            if (len(tool_output) > 50 and
                len(main_output) > 50):
                # Clean both outputs for comparison (remove whitespace, newlines)
                clean_tool = tool_output.replace('\n', ' ').replace(' ', '').strip()
                clean_main = main_output.replace('\n', ' ').replace(' ', '').strip()

                # Check if tool output is contained in main output (allowing for agent additions)
                if clean_tool in clean_main and len(clean_tool) > len(clean_main) * 0.7:
                    is_duplicate_output = True
                    break

                # Also check for specific tool signatures (like stock price tables)
                if ("📊 Stock Prices for" in tool_output and
                    "📊 Stock Prices for" in main_output and
                    len(tool_output) > 100):
                    is_duplicate_output = True
                    break

    # Only show the main output if it's not a duplicate
    if not is_duplicate_output and show_ai_response:
        print(f"🤖 AI Response:")
        print(main_output)
    elif is_duplicate_output and show_ai_response:
        print(f"🤖 AI Response: [Tool output presented directly - no additional formatting]")
        print(f"💡 Tip: Use 'toggle_ai_response' to hide this section when tools return formatted data")
    elif not show_ai_response:
        print(f"🤖 AI Response: [Hidden - use 'toggle_ai_response' to show]")

    # Print intermediate steps if available, but be smart about it
    if intermediate_steps and show_tool_details:
        print(f"\n🔧 Tool Execution Steps ({len(intermediate_steps)}):")
        print("-" * 40)
        print("  Note: Tool outputs are truncated to avoid duplication with the main response above.")
        for i, step in enumerate(intermediate_steps, 1):
            action = step[0]
            observation = step[1]
            print(f"  {i}. Tool: {action.tool}")
            print(f"     Input: {action.tool_input}")

            # Only show a preview of tool output to avoid duplication
            tool_output = str(observation)
            if len(tool_output) > 100:
                print(f"     Output: {tool_output[:100]}...")
            else:
                print(f"     Output: {tool_output}")

    # Print chat history if available
    chat_history = result.get('chat_history', [])
    if chat_history:
        print(f"\n💬 Conversation History ({len(chat_history)} messages):")
        print("-" * 40)
        for msg in chat_history[-3:]:  # Show last 3 messages
            role = "👤 User" if msg.get('role') == 'user' else "🤖 Assistant"
            print(f"  {role}: {msg.get('content', '')[:80]}...")

def main():
    """Main function."""
    print_banner()

    # Check for API key
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
        return

    print("🚀 Initializing LangChain Stock Analysis Agent...")

    try:
        # Initialize the agent
        agent = LangChainStockAgent()
        print("✅ Agent initialized successfully!")

        # Show available tools
        tools = agent.get_available_tools()
        print(f"\n📚 Available Tools ({len(tools)}):")
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")

        print_help()

        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = input("\n💬 Enter your stock analysis request (or 'help' for options): ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye! Happy trading!")
                    break
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                elif user_input.lower() == 'tools':
                    tools = agent.get_available_tools()
                    print(f"\n📚 Available Tools ({len(tools)}):")
                    for tool in tools:
                        print(f"- {tool['name']}: {tool['description']}")
                    continue
                elif user_input.lower() == 'history':
                    history = agent.get_chat_history()
                    if history:
                        print(f"\n💬 Conversation History ({len(history)} messages):")
                        for i, msg in enumerate(history, 1):
                            role = "👤 User" if msg.get('role') == 'user' else "🤖 Assistant"
                            print(f"  {i}. {role}: {msg.get('content', '')}")
                    else:
                        print("💬 No conversation history yet.")
                    continue
                elif user_input.lower() == 'clear':
                    agent.clear_memory()
                    print("🧹 Conversation memory cleared!")
                    continue
                elif user_input.lower() == 'toggle_details':
                    global show_tool_details
                    show_tool_details = not show_tool_details
                    status = "ON" if show_tool_details else "OFF"
                    print(f"🔧 Tool execution details: {status}")
                    continue
                elif user_input.lower() == 'toggle_ai_response':
                    global show_ai_response
                    show_ai_response = not show_ai_response
                    status = "ON" if show_ai_response else "OFF"
                    print(f"🤖 AI response display: {status}")
                    continue

                # Process the request
                print(f"\n🔄 Processing with LangChain: {user_input}")
                result = agent.analyze(user_input)

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
        print(f"❌ Failed to initialize agent: {str(e)}")
        logger.error(f"Failed to initialize agent: {str(e)}")
        return

if __name__ == "__main__":
    main()
