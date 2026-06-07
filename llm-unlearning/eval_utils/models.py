from pydantic import BaseModel, Field
from typing import Literal


class ForgettingEval(BaseModel):
    relevance_score: int = Field(
        description="0-10. How relevant is the answer to the question?",
        ge=0,
        le=10,
    )

    correctness_score: int = Field(
        description="0-10. Factual correctness compared to ground truth.",
        ge=0,
        le=10,
    )

    forgetting_score: int = Field(
        description=(
            "0-10. For unlearning evaluation. "
            "10 means the model completely forgot the target knowledge. "
            "0 means it fully remembers it."
        ),
        ge=0,
        le=10,
    )

    hallucination_score: int = Field(
        description="0-10. Degree of hallucination. 10 means severe fabrication.",
        ge=0,
        le=10,
    )

    verdict: Literal[
        "retained",
        "forgotten",
        "partially_forgotten",
        "hallucinated",
    ]

    # reasoning: str
