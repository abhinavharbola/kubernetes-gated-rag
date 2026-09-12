from openai import OpenAI
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from src.config import settings

nim_client = OpenAI(
    api_key=settings.nvidia_nim_api_key,
    base_url=settings.nim_base_url,
    timeout=settings.generation_timeout_seconds,
    max_retries=0,
)

groq_client = OpenAI(
    api_key=settings.groq_api_key,
    base_url=settings.groq_base_url,
    timeout=settings.generation_timeout_seconds,
    max_retries=0,
)

groq_client_secondary = OpenAI(
    api_key=settings.groq_api_key_secondary,
    base_url=settings.groq_base_url,
    timeout=settings.generation_timeout_seconds,
    max_retries=0,
)

# http_options.timeout is milliseconds in the google-genai SDK (unlike every
# other client here, which takes seconds) — previously unset entirely, so a
# slow embed_content call had no budget of its own and could run for however
# long the transport's own default allows.
gemini_client = genai.Client(
    api_key=settings.gemini_api_key,
    http_options=types.HttpOptions(timeout=int(settings.embedding_timeout_seconds * 1000)),
)

qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    timeout=settings.qdrant_timeout_seconds,
)



