from pydantic import BaseModel, Field, field_validator


class JudgeOutput(BaseModel):
    hallucination_score: float = Field(..., title="Hallucination score",
                                       description="Probability of answer being a hallucination "
                                                   "from 0.0 (none) to 1.0 (complete)")

    @field_validator('hallucination_score')
    @classmethod
    def check_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("hallucination_score must be between 0 and 1")
        return v
