from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from .prompt import ForgettingEvalPrompt
from .models import ForgettingEval
from .config import OpenAiConfig
from typing import Self
import warnings


class ForgettingLLMJudge:
    def __init__(self, openai_config: OpenAiConfig) -> None:
        self.judge_llm = ChatOpenAI(
            model=openai_config.model_name,
            base_url=openai_config.base_url,
            api_key=openai_config.api_key,
            temperature=0,
        )

        self.structured_judge = self.judge_llm.with_structured_output(
            ForgettingEval,
            method="json_schema",
        )

        self.fallback: Self | None = (
            self.__class__(openai_config.fallback)
            if openai_config.fallback is not None
            else None
        )

    def invoke_recursive(self, msgs: list[BaseMessage]) -> ForgettingEval:
        try:
            return self.structured_judge.invoke(msgs)  # pyright: ignore[reportAssignmentType]
        except Exception as e:
            if self.fallback is not None:
                warnings.warn(f"error: {e}")
                return self.fallback.invoke_recursive(msgs=msgs)
            raise Exception from e

    def invoke(
        self, question: str, ground_truth: str, prediction: str
    ) -> ForgettingEval:
        msgs = ForgettingEvalPrompt.format_messages(
            question=question, ground_truth=ground_truth, prediction=prediction
        )
        eval_result: ForgettingEval = self.invoke_recursive(msgs)  # pyright: ignore[reportAssignmentType]
        return eval_result
