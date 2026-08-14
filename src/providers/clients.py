from openai import OpenAI
from google import genai
from qdrant_client import QdrantClient

from src.config import settings

nim_client = OpenAI(
    api_key=settings.nvidia_nim_api_key,
    base_url=settings.nim_base_url,
    # generate_planner/generate_main retry once on the same provider before
    # moving to the next one in the chain (see src/providers/llm.py). Worst
    # case before that failover fires is roughly 2x this timeout, so 15s
    # here caps it around 30s rather than 60s.
    timeout=15.0,
    max_retries=0,
)

groq_client = OpenAI(
    api_key=settings.groq_api_key,
    base_url=settings.groq_base_url,
    timeout=15.0,
    max_retries=0,
)

gemini_client = genai.Client(api_key=settings.gemini_api_key)

qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    timeout=60.0,
)