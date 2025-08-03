import os
from typing import Optional

class Config:
    
    # Gemini API Configuration
    GEMINI_API_KEY: str = "AIzaSyCQ-AAImtpu_zjH8Obly8gk9vaWJopIgGo"  # Replace with your actual API key
    
    # Model Configuration
    GEMINI_MODEL: str = "gemini-1.5-flash"
    TEMPERATURE: float = 0.7
    
    # Application Settings
    DEFAULT_PROGRAM_DURATION: int = 1
    DEFAULT_MEALS_PER_DAY: int = 3
    DEFAULT_USERNAME: str = "username"
    
    # File Settings
    OUTPUT_DIRECTORY: str = "."
    FILE_ENCODING: str = "utf-8"
    
    # UI Settings
    CONSOLE_WIDTH: Optional[int] = None
    ENABLE_COLORS: bool = True
    
    @classmethod
    def get_gemini_api_key(cls) -> str:
        """Get Gemini API key from environment variable or config"""
        return os.getenv("GEMINI_API_KEY", cls.GEMINI_API_KEY)
    
    @classmethod
    def set_gemini_api_key(cls, api_key: str) -> None:
        """Set Gemini API key"""
        cls.GEMINI_API_KEY = api_key
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        if cls.get_gemini_api_key() == "YOUR_GEMINI_API_KEY_HERE":
            print("⚠️  Warning: Please set your Gemini API key in config.py or as GEMINI_API_KEY environment variable")
            return False
        return True


if os.getenv("GEMINI_API_KEY"):
    Config.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("GEMINI_API_KEY"):
        Config.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
except ImportError:
    pass  # python-dotenv not installed continue without it 