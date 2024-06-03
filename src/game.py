from dataclasses import dataclass, field
from typing import Optional
import random

@dataclass
class GameConfig:

    # word count
    word_count: Optional[int] = field(default=25, metadata={"help": "number of red words"})
    red_count: Optional[int] = field(default=9, metadata={"help": "number of red words"})
    blue_count: Optional[int] = field(default=8, metadata={"help": "number of blue words"})
    assassin_count: Optional[int] = field(default=1, metadata={"help": "number of assassin words"})
    innocent_count: Optional[int] = field(default=7, metadata={"help": "number of innocent words"})

    # word string
    red_string: Optional[str] = field(default="red", metadata={"help": "string used to represent red words"})
    blue_string: Optional[str] = field(default="blue", metadata={"help": "string used to represent blue words"})
    assassin_string: Optional[str] = field(default="black", metadata={"help": "string used to represent assassin words"})
    innocent_string: Optional[str] = field(default="white", metadata={"help": "string used to represent innocent words"})
    hidden_string: Optional[str] = field(default="?", metadata={"help": "string used to represent hidden words"})


class Card:
  def __init__(self, word, state, idx=-1, recto=False):
    self.idx = idx
    self.word = word
    self.state = state
    self.recto = recto

  def get_str(self, hide=False, hide_str='?'):
    return f"{self.idx}. {self.word} ({hide_str if hide and not self.recto else self.state})"


class Grid:
  def __init__(self, config, vocab):

    self.config = config
    self.cards = []

    self.fill_grid(vocab, self.config.red_string, self.config.red_count)
    self.fill_grid(vocab, self.config.blue_string, self.config.blue_count)
    self.fill_grid(vocab, self.config.assassin_string, self.config.assassin_count)
    self.fill_grid(vocab, self.config.innocent_string, self.config.innocent_count)

    random.shuffle(self.cards)

    for i, card in enumerate(self.cards):
      card.idx = i

  def fill_grid(self, vocab, state, count):
      words = random.sample(vocab, count)
      self.cards.extend([Card(word, state) for word in words])
      for w in words:
        vocab.remove(w)

  def get_str(self, hide=False):
      return '\n'.join(
          [card.get_str(hide, self.config.hidden_string) for card in self.cards]
      )