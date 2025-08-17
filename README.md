# Stock Analyzer AI - Modular System

An intelligent stock analysis system that uses OpenAI's GPT-5 API to parse natural language prompts and automatically execute technical analysis tools. The system features a clean, modular architecture with enhanced data display and comprehensive error handling.

## 🏗️ Architecture

The system is organized into focused, maintainable modules:

- **`config.py`** - Centralized configuration and environment variables
- **`data_fetcher.py`** - Stock data retrieval from Polygon.io API with smart fallbacks
- **`technical_indicators.py`** - Technical analysis calculations (RSI, Moving Averages, Bollinger Bands, MACD, Average Price)
- **`prompt_parser.py`** - OpenAI GPT-5 prompt parsing and execution planning
- **`stock_agent.py`** - Main agent that orchestrates the analysis workflow
- **`main.py`** - Interactive user interface with enhanced data display
- **`demo.py`** - Programmatic demonstration of the system
- **`test_agent.py` - Comprehensive test suite for all modules

## 🚀 Features

- **Natural Language Processing**: Use plain English to request stock analysis
- **GPT-5 Integration**: Latest OpenAI model for intelligent prompt parsing
- **Modular Design**: Clean separation of concerns for easy maintenance
- **Smart Data Fetching**: Automatic fallback from 1d to 5d for recent data
- **Enhanced Display**: Rich data previews with first/last rows for DataFrames
- **Comprehensive Analysis**: Multiple technical indicators with proper data validation
- **User-Friendly**: Interactive interface with helpful commands and examples
- **Robust Error Handling**: Graceful handling of API failures and insufficient data

## 📊 Available Tools

- **`fetch_stock_data`**: Get stock data for any ticker and time period
- **`calculate_rsi`**: Relative Strength Index calculation (requires sufficient data)
- **`calculate_moving_average`**: Simple Moving Average with configurable periods
- **`calculate_bollinger_bands`**: Volatility and trend analysis
- **`calculate_macd`**: Moving Average Convergence Divergence indicator
- **`calculate_average_price`**: Simple average of closing prices (works with any data amount)

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

- **OpenAI API Key**: For GPT-5 natural language prompt parsing
- **Polygon.io API Key**: For stock data retrieval

## 💻 Usage

### Interactive Mode (Recommended)

Run the interactive interface for custom prompts:

```bash
python main.py
```

**Available Commands:**
- `help` - Show available tools and examples
- `tools` - List all technical analysis tools
- `periods` - Show supported time periods
- `examples` - Display example prompts
- `quit` or `exit` - Exit the program

**Example Prompts:**
- "Get me Tesla stock for last month and calculate RSI"
- "Fetch Apple stock data for 3 months and calculate moving average with period 20"
- "Get Microsoft stock for last year and calculate Bollinger Bands"
- "Show me NVIDIA's last month stock prices, the RSI and the MACD"
- "Give me this week's stock price for Tesla and its average"

### Programmatic Usage

```python
from stock_agent import StockAnalysisAgent

# Initialize the agent
agent = StockAnalysisAgent()

# Analyze with custom prompt
result = agent.analyze("Get me NVIDIA stock for last 6 months and calculate RSI")

# Check results
if 'error' not in result:
    print("Analysis successful!")
    print(f"Results: {list(result['results'].keys())}")
    
    # Access specific results
    stock_data = result['results']['result_of_fetch_stock_data']
    rsi_values = result['results']['result_of_calculate_rsi']
else:
    print(f"Error: {result['error']}")
```

### Demo Mode

Run the demonstration script:

```bash
python demo.py
```

### Testing

Run the comprehensive test suite:

```bash
python test_agent.py
```

## 🔧 Adding New Features

### New Technical Indicators

1. Add the calculation method to `technical_indicators.py`
2. Update the tool descriptions in `prompt_parser.py`
3. Register the tool in `stock_agent.py`

### New Data Sources

1. Create a new data fetcher class
2. Implement the `fetch_stock_data` method
3. Update the main agent to use the new fetcher

## 📁 File Structure

```
stock_analyzer_ai/
├── config.py              # Configuration and environment variables
├── data_fetcher.py        # Stock data retrieval with smart fallbacks
├── technical_indicators.py # Technical analysis calculations
├── prompt_parser.py       # OpenAI GPT-5 prompt parsing
├── stock_agent.py         # Main analysis agent
├── main.py                # Interactive user interface with enhanced display
├── demo.py                # Demonstration script
├── test_agent.py          # Comprehensive test suite
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── .env                   # Environment variables (create from env_example.txt)
├── .gitignore             # Git ignore rules (protects API keys)
└── README.md              # This file
```

## 🧪 Testing

The system includes comprehensive tests for each module:

- Configuration module tests
- Data fetcher functionality with fallback logic
- Technical indicator calculations with data validation
- Prompt parser validation and error handling
- Main agent integration and workflow
- Modular import verification

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
4. **Empty RSI/MA Results**: Use longer periods (1mo+) for technical indicators that need more data
5. **OpenAI Empty Responses**: The system automatically handles token limits and model compatibility

### Getting Help

- Check the test output: `python test_agent.py`
- Review the demo: `python demo.py`
- Check logs for detailed error information
- Use the interactive help: `python main.py` then type `help`

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
