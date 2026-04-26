import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from main import build_chain, build_vectorstore, format_docs


class EvalCase(BaseModel):
    question: str
    expected_behaviour: str
    expected_source: Optional[str]
    notes: str


class EvalResult(BaseModel):
    question: str
    expected_behaviour: str
    actual_answer: str
    behaviour_correct: bool
    faithfulness_score: float
    faithfulness_reasoning: str
    relevance_score: float
    relevance_reasoning: str


class EvalSummary(BaseModel):
    total_cases: int
    behaviour_accuracy: float
    avg_faithfulness: float
    avg_relevance: float
    results: List[EvalResult]


REFUSE_PHRASES = [
    "not covered in the current controlled documents",
    "please consult your supervisor",
    "not covered",
    "no information",
]


class FaithfulnessJudgement(BaseModel):
    score: float
    reasoning: str


class RelevanceJudgement(BaseModel):
    score: float
    reasoning: str


def judge_faithfulness(question: str, answer: str, context: str) -> FaithfulnessJudgement:
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=512)
    structured = llm.with_structured_output(FaithfulnessJudgement)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are evaluating whether an AI answer is faithfully grounded in the provided source documents. "
            "Score from 0.0 (answer contradicts or fabricates information) to 1.0 (answer is fully supported by the documents). "
            "Be strict: if the answer includes any claim not found in the documents, reduce the score."
        )),
        ("human", (
            "Source documents:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer: {answer}\n\n"
            "Provide a faithfulness score and brief reasoning."
        )),
    ])

    return structured.invoke(prompt.format(context=context, question=question, answer=answer))


def judge_relevance(question: str, context: str) -> RelevanceJudgement:
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=512)
    structured = llm.with_structured_output(RelevanceJudgement)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are evaluating whether the retrieved document chunks are relevant to answering the question. "
            "Score from 0.0 (retrieved chunks are entirely irrelevant) to 1.0 (retrieved chunks directly address the question)."
        )),
        ("human", (
            "Question: {question}\n\n"
            "Retrieved chunks:\n{context}\n\n"
            "Provide a relevance score and brief reasoning."
        )),
    ])

    return structured.invoke(prompt.format(question=question, context=context))


def is_refusal(answer: str) -> bool:
    return any(phrase in answer.lower() for phrase in REFUSE_PHRASES)


def run_evals() -> EvalSummary:
    with open("evals/eval_dataset.json") as f:
        cases = [EvalCase(**c) for c in json.load(f)]

    vectorstore = build_vectorstore()
    chain = build_chain(vectorstore)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    results: List[EvalResult] = []

    for i, case in enumerate(cases):
        print(f"Running eval {i + 1}/{len(cases)}: {case.question[:60]}...")

        answer = chain.invoke(case.question)
        docs = retriever.invoke(case.question)
        context = format_docs(docs)

        refused = is_refusal(answer)
        behaviour_correct = (
            (case.expected_behaviour == "refuse" and refused) or
            (case.expected_behaviour == "answer" and not refused)
        )

        if case.expected_behaviour == "answer":
            faith = judge_faithfulness(case.question, answer, context)
            rel = judge_relevance(case.question, context)
        else:
            faith = FaithfulnessJudgement(score=1.0, reasoning="Refusal case — faithfulness not applicable.")
            rel = RelevanceJudgement(score=1.0, reasoning="Refusal case — relevance not applicable.")

        results.append(EvalResult(
            question=case.question,
            expected_behaviour=case.expected_behaviour,
            actual_answer=answer,
            behaviour_correct=behaviour_correct,
            faithfulness_score=faith.score,
            faithfulness_reasoning=faith.reasoning,
            relevance_score=rel.score,
            relevance_reasoning=rel.reasoning,
        ))

    answer_cases = [r for r in results if r.expected_behaviour == "answer"]

    summary = EvalSummary(
        total_cases=len(results),
        behaviour_accuracy=sum(r.behaviour_correct for r in results) / len(results),
        avg_faithfulness=sum(r.faithfulness_score for r in answer_cases) / len(answer_cases),
        avg_relevance=sum(r.relevance_score for r in answer_cases) / len(answer_cases),
        results=results,
    )

    return summary


def export_results(summary: EvalSummary, output_path: str):
    rows = [
        {
            "question": r.question,
            "expected_behaviour": r.expected_behaviour,
            "behaviour_correct": r.behaviour_correct,
            "faithfulness_score": r.faithfulness_score,
            "faithfulness_reasoning": r.faithfulness_reasoning,
            "relevance_score": r.relevance_score,
            "relevance_reasoning": r.relevance_reasoning,
            "actual_answer": r.actual_answer,
        }
        for r in summary.results
    ]

    df = pd.DataFrame(rows)
    meta = pd.DataFrame([{
        "total_cases": summary.total_cases,
        "behaviour_accuracy": f"{summary.behaviour_accuracy:.0%}",
        "avg_faithfulness": f"{summary.avg_faithfulness:.2f}",
        "avg_relevance": f"{summary.avg_relevance:.2f}",
    }])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Eval Results", index=False)
        meta.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\nResults written to {output_path}")


def main():
    print("Running evaluation suite...\n")
    summary = run_evals()

    print(f"\n--- Eval Summary ---")
    print(f"Total cases:        {summary.total_cases}")
    print(f"Behaviour accuracy: {summary.behaviour_accuracy:.0%}")
    print(f"Avg faithfulness:   {summary.avg_faithfulness:.2f}")
    print(f"Avg relevance:      {summary.avg_relevance:.2f}")

    Path("output").mkdir(exist_ok=True)
    export_results(summary, "output/eval_results.xlsx")


if __name__ == "__main__":
    main()
