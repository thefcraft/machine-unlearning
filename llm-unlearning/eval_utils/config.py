import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, SecretStr
from typing import Self, IO
from dotenv import load_dotenv


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    @classmethod
    def from_env(cls) -> Self:
        return cls()  # pyright: ignore[reportCallIssue]

    @classmethod
    def from_dotenv(
        cls,
        dotenv_path: os.PathLike[str] | str | None = None,
        stream: IO[str] | None = None,
        verbose: bool = False,
        override: bool = False,
        interpolate: bool = True,
        encoding: str | None = "utf-8",
    ) -> Self:
        if dotenv_path is None:
            dotenv_path = ".env"
        if not os.path.exists(dotenv_path):
            raise FileNotFoundError(dotenv_path)
        load_dotenv(
            dotenv_path=dotenv_path,
            stream=stream,
            verbose=verbose,
            override=override,
            interpolate=interpolate,
            encoding=encoding,
        )
        return cls.from_env()


class OpenAiConfig(BaseModel):
    model_name: str
    base_url: str
    api_key: SecretStr
    fallback: Self | None = None


class Config(BaseConfig):
    openai: OpenAiConfig
