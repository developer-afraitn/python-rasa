from typing import Any, List, Text, Dict, Optional

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.components import Component
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer

from hazm import WordTokenizer, Normalizer

class HazmTokenizer(Tokenizer):
    """Tokenizer for Persian language using Hazm library."""

    defaults = {
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
        "token_pattern": None,
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        self.tokenizer = WordTokenizer()
        self.normalizer = Normalizer()

    def train(
        self, training_data: TrainingData, *args: Any, **kwargs: Any
    ) -> None:
        pass  # No training needed for tokenizer

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = self.normalizer.normalize(message.get(attribute))
        words = self.tokenizer.tokenize(text)

        tokens = []
        offset = 0
        for word in words:
            start = text.find(word, offset)
            end = start + len(word)
            token = Token(word, start)
            tokens.append(token)
            offset = end

        if not tokens:
            tokens = [Token(text, 0)]

        return self.add_cls_token(tokens)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        for example in training_data.training_examples:
            for attribute in self._attributes():
                if example.get(attribute):
                    example.set("tokens", self.tokenize(example, attribute))
        return training_data