import os
import tempfile

_TEST_ENV = {
    "NVIDIA_NIM_API_KEY": "test-nim-key",
    "GROQ_API_KEY": "test-groq-key",
    "GROQ_API_KEY_SECONDARY": "test-groq-secondary-key",
    "GEMINI_API_KEY": "test-gemini-key",
    "QDRANT_URL": "https://localhost:6333",
    "QDRANT_API_KEY": "test-qdrant-key",
    "GUARDRAIL_TIMEOUT_SECONDS": "3",
    "GUARDRAIL_CIRCUIT_FAILURE_THRESHOLD": "2",
    "GUARDRAIL_SKIP_NEMOGUARD_SAFETY": "false",
    "GUARDRAIL_SKIP_NEMOGUARD_TOPIC": "true",
    "PROVIDER_CIRCUIT_FAILURE_THRESHOLD": "2",
    "NO_CONTEXT_CACHE_TTL_SECONDS": "3600",
    "CORPUS_VERSION": "1",
    "HISTORY_MAX_MESSAGES": "6",
    "GENERATION_CONTEXT_CHUNKS": "5",
}

for key, value in _TEST_ENV.items():
    os.environ[key] = value

os.environ.pop("LOGFIRE_TOKEN", None)
os.environ["CACHE_DIR"] = tempfile.mkdtemp(prefix="k8s-rag-test-")
