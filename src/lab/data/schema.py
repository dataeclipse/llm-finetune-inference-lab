from pydantic import BaseModel, Field


class SQLExample(BaseModel):
    question: str
    context: str
    sql: str
    complexity: str = ""
    domain: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class DatasetSummary(BaseModel):
    raw_examples: int
    cleaned_examples: int
    train_examples: int
    val_examples: int
    test_examples: int
    dropped: dict[str, int] = Field(default_factory=dict)
