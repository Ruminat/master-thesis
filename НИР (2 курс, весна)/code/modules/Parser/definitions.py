from typing import Optional

class MeCabToken:
  def __init__(self, token: str, partOfSpeech: Optional[str] = None):
    self.token = token
    self.partOfSpeech = partOfSpeech
