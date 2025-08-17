"""
Setup script for the AI-Powered Stock Analysis Agent

This script helps users set up the environment and test the system.
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")

    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from template."""
    print("\n🔑 Setting up environment file...")

    if os.path.exists('.env'):
        print("✅ .env file already exists")
        return True

    if os.path.exists('env_example.txt'):
        try:
            shutil.copy('env_example.txt', '.env')
            print("✅ .env file created from template")
            print("   Please edit .env and add your API keys")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("❌ env_example.txt not found")
        return False

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API keys...")

    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv('OPENAI_API_KEY')
    polygon_key = os.getenv('POLYGON_API_KEY')

    if openai_key and openai_key != 'your_openai_api_key_here':
        print("✅ OPENAI_API_KEY configured")
    else:
        print("❌ OPENAI_API_KEY not configured")
        print("   Please add your OpenAI API key to .env file")

    if polygon_key and polygon_key != 'your_polygon_api_key_here':
        print("✅ POLYGON_API_KEY configured")
    else:
        print("❌ POLYGON_API_KEY not configured")
        print("   Please add your Polygon.io API key to .env file")

    return bool(openai_key and openai_key != 'your_openai_api_key_here')

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")

    try:
        subprocess.check_call([sys.executable, "test_agent.py"])
        print("✅ Tests completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 AI-Powered Stock Analysis Agent - Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        return False

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Dependencies installation failed")
        return False

    # Create environment file
    if not create_env_file():
        print("\n❌ Setup failed: Environment file creation failed")
        return False

    # Check API keys
    if not check_api_keys():
        print("\n⚠️  Setup completed but API keys need configuration")
        print("   Please edit .env file and add your API keys")
        print("   Then run: python test_agent.py")
        return True

    # Run tests
    if not run_tests():
        print("\n⚠️  Setup completed but tests failed")
        print("   Please check your configuration and try again")
        return False

    print("\n🎉 Setup completed successfully!")
    print("\n🚀 Next steps:")
    print("   1. Try the demo: python demo.py")
    print("   2. Use in your code: from stock_agent import StockAnalysisAgent")
    print("   3. Run your own analysis: agent.analyze('Your prompt here')")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
