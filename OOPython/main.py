import abc
from abc import abstractclassmethod


class Card():
    insure = False
    def __init__(self, rank, suit):
        self.suit = suit
        self.rank = rank
        self.hard, self.soft = self._points()

    def __repr__(self):
            return "{__class__.__name__}(suit={suit!r}, rank={rank!r})".format(
                __class__=self.__class__, **self.__dict__)

    def __str__(self):
        return "{rank}{suit}".format(**self.__dict__)


class NumberCard(Card):
    def _points(self):
        return int(self.rank), int(self.rank)


class Hand:
    def __init__(self, dealer_card, *cards):
        self.dealer_card = dealer_card
        self.cards = list(cards)

    def __str__(self):
        return ", ".join(map(str, self.cards))

    def __repr__(self):
        return "{__class__.__name__}({dealer_card!r}, {_cards_str})".format(
            __class__=self.__class__,
            _cards_str=", ".join(map(repr, self.cards)),
            **self.__dict__)

x = NumberCard('2', '♣')

print(x.__repr__())
print(x)
print(x.__dict__)


class Car(object):
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def __str__(self):
        return "{name} {color}".format(**self.__dict__)

    def __repr__(self):
        return "{__class__.__name__}(name={name!r}, color={color!r})".format(
            __class__=self.__class__, **self.__dict__)


class BlackJackCard:
    def __init__(self, rank, suit, hard, soft):

        self.rank = rank
        self.suit = suit
        self.hard = hard
        self.soft = soft

    def __lt__(self, other):
        if not isinstance(other, BlackJackCard):
            return NotImplemented
        return self.rank < other.rank


    def __le__(self, other):
        try:
            return self.rank <= other.rank
        except AttributeError:
            return NotImplemented


    def __gt__(self, other):
        if not isinstance(other, BlackJackCard):
            return NotImplemented
        return self.rank > other.rank


    def __ge__(self, other):
        if not isinstance(other, BlackJackCard):
            return NotImplemented
        return self.rank >= other.rank


    def __eq__(self, other):
        if not isinstance(other, BlackJackCard):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit


    def __ne__(self, other):
        if not isinstance(other, BlackJackCard):
            return NotImplemented
        return self.rank != other.rank and self.suit != other.suit


    def __str__(self):
        return "{rank}{suit}".format(**self.__dict__)


class Parent:
    def __init__(self, *children):
        self.children = list(children)
        for child in self.children:
            child.parent = self

    def __del__(self):
        print("Removing {__class__.__name__} {id:d}".format(__class__=self.__class__, id=id(self)))


class Child:
    def __del__(self):
        print("Removing {__class__.__name__} {id:d}".format(__class__=self.__class__, id=id(self)))
