"""
Configuration module for Stock Analysis AI

Centralizes all environment variables, API keys, and configuration settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5-mini')  # Updated to use GPT-5
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '5000'))
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '1'))

# Polygon.io API configuration
POLYGON_BASE_URL = "https://api.polygon.io"

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
