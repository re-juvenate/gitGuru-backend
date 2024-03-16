from pydantic import BaseModel

class Repo(BaseModel):
    url: str
