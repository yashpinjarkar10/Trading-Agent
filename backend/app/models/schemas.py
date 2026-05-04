from pydantic import BaseModel, Field, field_validator
from typing import Optional


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    period: Optional[str] = "1y"
    days_back: Optional[int] = Field(7, ge=1, le=90)
    max_articles: Optional[int] = Field(50, ge=1, le=200)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    # Bug #9: thread_id is REQUIRED — no shared "default" thread across users.
    # Must be unique per (user, session). Min length 8 to discourage collisions.
    thread_id: str = Field(..., min_length=8, max_length=128)

    @field_validator("thread_id")
    @classmethod
    def _no_default_thread(cls, v: str) -> str:
        if v.strip().lower() in {"default", "test", "shared", ""}:
            raise ValueError(
                "thread_id must be unique per user/session "
                "(reserved values like 'default' are not allowed)"
            )
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    thread_id: str
