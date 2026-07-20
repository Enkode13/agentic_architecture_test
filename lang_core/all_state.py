from pydantic import BaseModel, Field
from typing import Annotated
import operator

class State(BaseModel):
    query: str = ""
    input: str = ""
    answer: str = ""
    tool: str = ""
    scratchpad: Annotated[list[dict], operator.add] = Field(default_factory=list)
    end: bool = False
    steps: Annotated[int, operator.add] = 0