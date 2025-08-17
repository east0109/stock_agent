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

Guidelines:
1. Always fetch stock data first before calculating indicators
2. Use appropriate time periods: 1mo+ for RSI/MA/BB, 3mo+ for MACD
3. For "latest" requests, use 5d period to ensure data availability
4. Provide clear explanations of your analysis
5. Handle errors gracefully and suggest alternatives

When asked to analyze stocks, follow this workflow:
1. Fetch the required stock data
2. Calculate the requested technical indicators
3. Provide insights and analysis
4. Suggest additional analysis if relevant

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
