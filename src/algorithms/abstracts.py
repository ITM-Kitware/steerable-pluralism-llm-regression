from abc import ABC, abstractmethod
from typing import  Dict

class ADM(ABC):
    @abstractmethod
    def choose_response(self,
                        prompt: str,
                        responses: list,
                        alignment_target: Dict):
        pass
