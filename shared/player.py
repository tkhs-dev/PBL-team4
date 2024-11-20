import abc
from abc import ABCMeta

from shared.rule import Direction


class IPlayer(metaclass=ABCMeta):
    @abc.abstractmethod
    def on_start(self, game_state):
        raise NotImplementedError

    @abc.abstractmethod
    def on_end(self, game_state):
        raise NotImplementedError

    @abc.abstractmethod
    def on_move(self, game_state) -> Direction:
        raise NotImplementedError