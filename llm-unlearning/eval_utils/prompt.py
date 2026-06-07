from langchain_core.prompts import ChatPromptTemplate


class ForgettingEvalPrompt:
    SYSTEM: str = """
You are evaluating an LLM unlearning experiment.

Definitions:

RETAINED:
- Answer matches ground truth knowledge.

FORGOTTEN:
- Model correctly states it does not know.
- Or refuses because knowledge has been removed.

PARTIALLY_FORGOTTEN:
- Some correct information remains.

HALLUCINATED:
- Information is fabricated and unsupported.

Score:
- correctness_score: 0-10
- forgetting_score: 0-10
- hallucination_score: 0-10
- relevance_score: 0-10

Return structured output only.
"""
    HUMAN: str = """
Question:
{question}

Ground Truth:
{ground_truth}

Model Output:
{prediction}
"""
    PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            ("human", HUMAN),
        ]
    )

    @classmethod
    def format_messages(cls, question: str, ground_truth: str, prediction: str):
        return cls.PROMPT.format_messages(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
