import asyncio

import pandas as pd
from openai import AsyncOpenAI

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
    SemanticSimilarity,
)

from eval.dataset import load_eval_set
from src.config import settings
from src.retrieval.embeddings import embed_texts
from src.providers.llm import generate_main
from src.graph import ANSWER_SYSTEM_PROMPT

# Gemini's OpenAI-compatibility endpoint — lets the judge use AsyncOpenAI
# the same way build_judge used to point at Groq, just with a different
# base_url/key. https://ai.google.dev/gemini-api/docs/openai
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Gemini's free tier is far tighter than Groq's (roughly 15 RPM for
# gemini-3.5-flash at time of writing) and this script fires every eval
# row's metrics concurrently via asyncio.gather — uncapped, that bursts well
# past 15 RPM even for the 8-row starter set. This semaphore caps how many
# rows are scored concurrently; each row still fires several judge/embedding
# calls internally, so keep this low rather than sized to the row count.
_JUDGE_CONCURRENCY = 2


class GeminiRagasEmbedding(BaseRagasEmbedding):
    def embed_text(self, text: str, **kwargs) -> list[float]:
        return embed_texts([text], task_type="SEMANTIC_SIMILARITY")[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def build_judge():
    # deliberately a different model family from both live chains (Groq
    # gpt-oss, NIM llama) so the judge never grades a model from its own
    # family. Gemini is otherwise only used for embeddings in this project
    # (src/retrieval/embeddings.py) — this is its one other role.
    client = AsyncOpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_BASE_URL)
    return llm_factory(settings.gemini_eval_judge_model, client=client)


def _generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    result = generate_main(
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ]
    )
    return result.content


async def _score_row(row: dict, llm, embeddings) -> dict:
    answer = row.get("answer") or _generate_answer(row["question"], row["retrieved_contexts"])

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecisionWithReference(llm=llm)
    context_recall = ContextRecall(llm=llm)
    context_entity_recall = ContextEntityRecall(llm=llm)
    semantic_similarity = SemanticSimilarity(embeddings=embeddings)

    faithfulness_result, relevancy_result, precision_result, recall_result, entity_recall_result, similarity_result = (
        await asyncio.gather(
            faithfulness.ascore(
                user_input=row["question"], response=answer, retrieved_contexts=row["retrieved_contexts"]
            ),
            answer_relevancy.ascore(user_input=row["question"], response=answer),
            context_precision.ascore(
                user_input=row["question"],
                response=answer,
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["ground_truth"],
            ),
            context_recall.ascore(
                user_input=row["question"],
                response=answer,
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["ground_truth"],
            ),
            context_entity_recall.ascore(
                retrieved_contexts=row["retrieved_contexts"], reference=row["ground_truth"]
            ),
            semantic_similarity.ascore(reference=row["ground_truth"], response=answer),
        )
    )

    return {
        "question": row["question"],
        "answer": answer,
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
        "context_precision": precision_result.value,
        "context_recall": recall_result.value,
        "context_entity_recall": entity_recall_result.value,
        "semantic_similarity": similarity_result.value,
    }


async def _run_eval_async() -> pd.DataFrame:
    records = load_eval_set()
    llm = build_judge()
    embeddings = GeminiRagasEmbedding()
    semaphore = asyncio.Semaphore(_JUDGE_CONCURRENCY)

    async def _score_row_limited(record: dict) -> dict:
        async with semaphore:
            return await _score_row(record, llm, embeddings)

    rows = await asyncio.gather(*(_score_row_limited(record) for record in records))
    return pd.DataFrame(rows)


def run_eval() -> None:
    results_df = asyncio.run(_run_eval_async())
    print(results_df.describe())
    results_df.to_csv("eval/results.csv", index=False)


if __name__ == "__main__":
    run_eval()