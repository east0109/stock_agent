"""
LangChain-based Stock Analysis Agent

This module provides a LangChain-powered agent for stock analysis,
with real tool execution and proper function calling.
"""
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_tools import get_all_tools
from config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

class LangChainStockAgent:
    """LangChain-based stock analysis agent using function calling with tools."""

    def __init__(self):
        """Initialize the LangChain agent."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")

        # Initialize OpenAI LLM with streaming completely disabled
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=1,  # Set to None to override default 0.7
            streaming=False,
            callbacks=None
        )

        # Get tools
        self.tools = get_all_tools()

        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a stock analysis assistant. You help users analyze stocks by fetching data and calculating technical indicators.

Available tools:
- fetch_stock_data: Get stock data for a ticker and time period
- calculate_rsi: Calculate Relative Strength Index
- calculate_moving_average: Calculate Simple Moving Average
- calculate_bollinger_bands: Calculate Bollinger Bands
- calculate_macd: Calculate MACD indicator
- calculate_average_price: Calculate average closing price
- display_stock_prices: Display stock prices in a formatted table with dates and changes. Can show 'open', 'close', or 'both' prices.

Guidelines:
1. Always fetch stock data first before calculating indicators
2. Use appropriate time periods: 1mo+ for RSI/MA/BB, 3mo+ for MACD
3. For "latest" requests, use 5d period to ensure data availability
4. Provide clear explanations of your analysis
5. Handle errors gracefully and suggest alternatives

IMPORTANT RULES FOR PRICE DISPLAY REQUESTS:
- When a user asks to "display prices", "show prices", "list prices", or similar, ALWAYS use the display_stock_prices tool
- Do NOT ask clarifying questions about format - just execute the tool and show the actual data
- The display_stock_prices tool will automatically format the data nicely with dates, prices, and changes
- If no specific period is mentioned, default to "1mo" (1 month)
- For price type options:
  * "close" or "closing" → shows only closing prices (default)
  * "open" → shows only opening prices
  * "both" or "open_close" → shows both open and close prices
  * If user asks for "open and close" or "everyday's open and closing", use "both"

CRITICAL: DO NOT REFORMAT TOOL RESPONSES
- When tools return already-formatted data (like tables, charts, or structured output), present them DIRECTLY to the user
- Do NOT add extra formatting, headers, or explanations that duplicate what the tool already provided
- Only add context when the tool returns raw data that needs interpretation
- The goal is to show the user exactly what the tool returned, not a reformatted version
- NEVER repeat or restate what the tool already said - just show the tool's output as-is
- If a tool returns a complete, well-formatted response, your job is done - just present it

EXAMPLE OF CORRECT BEHAVIOR:
User: "Show me Tesla's stock prices for the last 30 days"
Agent: [Executes display_stock_prices tool]
Tool returns: "📊 Stock Prices for TSLA (1mo) - Close Prices\nDate Range: 2025-07-18 to 2025-08-15\n..."
Agent response: [Tool output exactly as returned - NO reformatting, NO duplication]

EXAMPLE OF WRONG BEHAVIOR:
User: "Show me Tesla's stock prices for the last 30 days"
Agent: [Executes display_stock_prices tool]
Tool returns: "📊 Stock Prices for TSLA (1mo) - Close Prices\nDate Range: 2025-07-18 to 2025-08-15\n..."
Agent response: "Here are Tesla's stock prices for the last 30 days:\n\n📊 Stock Prices for TSLA (1mo) - Close Prices\nDate Range: 2025-07-18 to 2025-08-15\n..." [WRONG - duplicates the tool output]

When asked to analyze stocks, follow this workflow:
1. Fetch the required stock data
2. Calculate the requested technical indicators
3. Provide insights and analysis
4. Suggest additional analysis if relevant

When asked to display prices:
1. Use the display_stock_prices tool immediately
2. Choose the appropriate price type based on user request
3. Present the formatted price data DIRECTLY to the user (no reformatting)
4. Offer to calculate additional indicators if relevant

IMPORTANT: Use the available tools to perform actual analysis. Don't just describe what you would do - execute the tools and provide real results."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a stock analysis request using LangChain with function calling.

        Args:
            prompt (str): Natural language request

        Returns:
            Dict containing results and execution details
        """
        try:
            logger.info(f"Processing prompt with LangChain: {prompt}")

            # Execute the agent
            result = self.agent_executor.invoke({"input": prompt})

            logger.info(f"LangChain execution completed: {result.get('output', 'No output')}")

            return {
                "success": True,
                "output": result.get('output', 'No output'),
                "intermediate_steps": result.get('intermediate_steps', []),
                "chat_history": self.get_chat_history(),
                "prompt": prompt
            }

        except Exception as e:
            logger.error(f"Error in LangChain analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get available tools and their descriptions."""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in self.memory.chat_memory.messages
        ]

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
