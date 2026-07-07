import os
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')
OPEN_AI_API_URL = os.getenv('OPEN_AI_API_URL')

AZURE_API_VERSION = '2024-10-21'

VECTOR_DIMENSIONS = 1536