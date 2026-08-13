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


class GeminiRagasEmbedding(BaseRagasEmbedding):
    def embed_text(self, text: str, **kwargs) -> list[float]:
        return embed_texts([text], task_type="SEMANTIC_SIMILARITY")[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def build_judge():
    # deliberately separate from whatever provider is serving live traffic,
    # so eval runs never compete with user-facing rate limits.
    client = AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    return llm_factory(settings.groq_eval_judge_model, client=client)


def _generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    result = generate_main(
        [
            {
                "role": "system",
                "content": "Answer the Kubernetes question using ONLY the provided context. "
                "If the context doesn't fully cover the question, say what's missing "
                "rather than guessing.",
            },
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

    rows = await asyncio.gather(*(_score_row(record, llm, embeddings) for record in records))
    return pd.DataFrame(rows)


def run_eval() -> None:
    results_df = asyncio.run(_run_eval_async())
    print(results_df.describe())
    results_df.to_csv("eval/results.csv", index=False)


if __name__ == "__main__":
    run_eval()