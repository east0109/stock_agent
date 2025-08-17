# Stock Analyzer AI - LangChain Agent System

An intelligent stock analysis system powered by LangChain and OpenAI's GPT models. The system uses natural language processing to automatically execute technical analysis tools through an intelligent agent that can understand complex requests and provide comprehensive stock analysis.

## 🏗️ Architecture

The system is built around a modern LangChain agent architecture:

- **`langchain_agent.py`** - Core LangChain agent with OpenAI integration
- **`langchain_tools.py`** - Custom tools for stock analysis (RSI, Moving Averages, Bollinger Bands, MACD, etc.)
- **`data_fetcher.py`** - Stock data retrieval from Polygon.io API
- **`technical_indicators.py`** - Technical analysis calculations
- **`main_langchain.py`** - Interactive user interface for the LangChain agent
- **`config.py`** - Centralized configuration and environment variables

## 🚀 Features

- **Natural Language Processing**: Use plain English to request stock analysis
- **LangChain Agent**: Intelligent tool orchestration and execution
- **OpenAI Integration**: Latest GPT models for understanding complex requests
- **Smart Data Fetching**: Automatic data retrieval with intelligent period selection
- **Enhanced Display**: Rich, formatted tables for stock prices and analysis results
- **Comprehensive Analysis**: Multiple technical indicators with proper data validation
- **User-Friendly Interface**: Interactive commands with helpful tips and examples
- **Robust Error Handling**: Graceful handling of API failures and insufficient data
- **Smart Output Control**: Eliminates duplicate output with intelligent detection

## 📊 Available Tools

The LangChain agent has access to these specialized tools:

- **`fetch_stock_data`**: Get stock data for any ticker and time period
- **`display_stock_prices`**: Show formatted tables of daily prices (open, close, or both)
- **`calculate_rsi`**: Relative Strength Index calculation with intelligent period selection
- **`calculate_moving_average`**: Simple Moving Average with configurable periods
- **`calculate_bollinger_bands`**: Volatility and trend analysis
- **`calculate_macd`**: Moving Average Convergence Divergence indicator
- **`calculate_average_price`**: Average of closing prices over specified periods

## ⏰ Supported Time Periods

- **1d, 5d** - Recent data (weekends/holidays excluded)
- **1mo, 3mo, 6mo** - Short to medium term analysis
- **1y, 2y, 5y, 10y** - Long term analysis
- **ytd, max** - Year-to-date and maximum available data

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd stock_analyzer_ai
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp env_example.txt .env
# Edit .env with your API keys
```

## 🔑 Required API Keys

- **OpenAI API Key**: For GPT model integration and natural language understanding
- **Polygon.io API Key**: For stock data retrieval

## 💻 Usage

### Interactive Mode (Recommended)

Run the interactive LangChain agent interface:

```bash
python main_langchain.py
```

**Available Commands:**
- `help` - Show available tools and examples
- `tools` - List all technical analysis tools
- `history` - Show conversation history
- `clear` - Clear conversation memory
- `toggle_details` - Toggle tool execution details on/off
- `toggle_ai_response` - Toggle AI response display on/off
- `quit` or `exit` - Exit the program

**Example Prompts:**
- "Show me Tesla's stock prices for the last 30 days"
- "Display Apple's open and close prices for the past month"
- "Get NVIDIA stock data and calculate RSI for the last 3 months"
- "Calculate moving average with period 20 for Microsoft stock over 6 months"
- "Show me Tesla's stock prices for each day for the last 30 days"

### Programmatic Usage

```python
from langchain_agent import LangChainStockAgent

# Initialize the agent
agent = LangChainStockAgent()

# Analyze with custom prompt
result = agent.analyze("Show me Tesla's stock prices for the last month")

# Check results
if result.get('success', False):
    print("Analysis successful!")
    print(f"Output: {result.get('output', 'No output')}")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
```

## 🔧 Smart Features

### Intelligent Period Selection
- **RSI/MA/BB**: Automatically defaults to 1 month if no period specified
- **MACD**: Intelligently suggests 3+ months for optimal calculation
- **Data Validation**: Checks if requested periods have sufficient data

### Smart Output Control
- **Duplicate Detection**: Automatically identifies when agent repeats tool output
- **Clean Display**: Eliminates triple output issues with intelligent formatting
- **User Control**: Toggle commands to customize what information is shown

### Flexible Input Parsing
- **Multiple Formats**: Handles pipe (|), colon (:), comma (,), and space-separated arguments
- **Smart Defaults**: Automatically fills in missing parameters
- **Error Recovery**: Graceful handling of malformed requests

## 📁 File Structure

```
stock_analyzer_ai/
├── langchain_agent.py      # Core LangChain agent implementation
├── langchain_tools.py      # Custom tools for stock analysis
├── main_langchain.py       # Interactive user interface
├── data_fetcher.py         # Stock data retrieval
├── technical_indicators.py # Technical analysis calculations
├── config.py               # Configuration and environment variables
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── .env                    # Environment variables (create from env_example.txt)
├── .gitignore              # Git ignore rules (protects API keys)
└── README.md               # This file
```

## 🧪 Testing

Test individual tools and the complete system:

```bash
# Test specific tools
python -c "from langchain_tools import get_all_tools; tools = get_all_tools(); print(f'Found {len(tools)} tools')"

# Test the agent
python main_langchain.py
# Then try: "Show me Tesla's stock prices for the last week"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file contains valid API keys
2. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
3. **Data Fetching Failures**: Check your Polygon.io API key and internet connection
4. **Empty Results**: Use longer periods (1mo+) for technical indicators that need more data
5. **Duplicate Output**: Use `toggle_ai_response` to hide redundant AI responses

### Getting Help

- Check the interactive help: `python main_langchain.py` then type `help`
- Use toggle commands to control output: `toggle_details` and `toggle_ai_response`
- Review tool execution steps for debugging
- Check logs for detailed error information

## 🔮 Future Enhancements

- Additional technical indicators (Fibonacci retracements, Stochastic oscillator)
- Multiple data source support (Yahoo Finance, Alpha Vantage)
- Real-time data streaming
- Portfolio analysis tools
- Machine learning-based predictions
- Web interface
- Mobile app support
- Advanced charting and visualization
- Historical performance tracking
- Enhanced natural language understanding for complex queries
