# Stock Analyzer AI - Modular System

An intelligent stock analysis system that uses OpenAI's API to parse natural language prompts and automatically execute technical analysis tools. The system has been refactored into a modular architecture for better maintainability and extensibility.

## 🏗️ Architecture

The system is now organized into several focused modules:

- **`config.py`** - Centralized configuration and environment variables
- **`data_fetcher.py`** - Stock data retrieval from Polygon.io API
- **`technical_indicators.py`** - Technical analysis calculations (RSI, Moving Averages, Bollinger Bands, MACD)
- **`prompt_parser.py`** - OpenAI prompt parsing and execution planning
- **`stock_agent.py`** - Main agent that orchestrates the analysis
- **`main.py`** - Interactive user interface for custom prompts
- **`demo.py`** - Programmatic demonstration of the system
- **`test_agent.py`** - Test suite

## 🚀 Features

- **Natural Language Processing**: Use plain English to request stock analysis
- **Modular Design**: Clean separation of concerns for easy maintenance
- **Extensible**: Easy to add new technical indicators or data sources
- **Comprehensive Analysis**: Supports multiple technical indicators
- **User-Friendly**: Interactive interface for custom prompts
- **Robust Error Handling**: Graceful handling of API failures and invalid inputs

## 📊 Available Tools

- **Data Fetching**: Get stock data for any ticker and time period
- **RSI**: Relative Strength Index calculation
- **Moving Averages**: Simple Moving Average with configurable periods
- **Bollinger Bands**: Volatility and trend analysis
- **MACD**: Moving Average Convergence Divergence indicator

## ⏰ Supported Time Periods

- 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock_analyzer_ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env_example.txt .env
# Edit .env with your API keys
```

## 🔑 Required API Keys

- **OpenAI API Key**: For natural language prompt parsing
- **Polygon.io API Key**: For stock data retrieval

## 💻 Usage

### Interactive Mode (Recommended)

Run the interactive interface for custom prompts:

```bash
python main.py
```

Example prompts:
- "Get me Tesla stock for last month and calculate RSI"
- "Fetch Apple stock data for 3 months and calculate moving average with period 20"
- "Get Microsoft stock for last year and calculate Bollinger Bands"

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
else:
    print(f"Error: {result['error']}")
```

### Demo Mode

Run the demonstration script:

```bash
python demo.py
```

### Testing

Run the test suite:

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
├── data_fetcher.py        # Stock data retrieval
├── technical_indicators.py # Technical analysis calculations
├── prompt_parser.py       # OpenAI prompt parsing
├── stock_agent.py         # Main analysis agent
├── main.py                # Interactive user interface
├── demo.py                # Demonstration script
├── test_agent.py          # Test suite
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── .env                   # Environment variables (create from env_example.txt)
└── README.md              # This file
```

## 🧪 Testing

The system includes comprehensive tests for each module:

- Configuration module tests
- Data fetcher functionality
- Technical indicator calculations
- Prompt parser validation
- Main agent integration
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

### Getting Help

- Check the test output: `python test_agent.py`
- Review the demo: `python demo.py`
- Check logs for detailed error information

## 🔮 Future Enhancements

- Additional technical indicators (Fibonacci retracements, Stochastic oscillator)
- Multiple data source support (Yahoo Finance, Alpha Vantage)
- Real-time data streaming
- Portfolio analysis tools
- Machine learning-based predictions
- Web interface
- Mobile app support
