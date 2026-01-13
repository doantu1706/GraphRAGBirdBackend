import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # User Agent giả lập trình duyệt để tránh bị chặn khi cào dữ liệu
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"