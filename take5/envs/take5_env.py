import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class Take5Env(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, sides=2, debug=True):
    self.sides = sides
    self.debug = debug

    self.max_card = 104
    self.max_hand = 10
    self.table_rows = 4
    self.points = np.ones(self.max_card+1)
    self.points[0] = 0
    self.round = 0
    for c in range(5, self.max_card, 5):
        self.points[c] = 2
    for c in range(10, self.max_card, 10):
        self.points[c] = 3
    for c in range(11, self.max_card, 11):
        self.points[c] = 5

  def _draw_card(self, shape):
    drawn = self.deck[:np.prod(shape)].reshape(shape)
    self.deck = self.deck[np.prod(shape):]
    return drawn

  def _reset_row(self, row, player):
    self.penalties[player] -= self.table_points.sum(-1)[row]
    self.accum_penalties[player] -= self.table_points.sum(-1)[row]
    self.table[row] = 0

  def step(self, action):
    played_cards = []
    for h, hand in enumerate(self.hands):
      if h==0:
        a = action
      else:
        a = hand.argmax()
      played_cards.append((hand[a], h))
      hand[a] = 0
    played_cards = np.array(played_cards, dtype=[('card', int), ('player', int)])
    played_cards = np.sort(played_cards, order='card')
    self.penalties = np.zeros(self.sides)

    for card, player in played_cards:
      diff = card - self.table.max(-1)
      if np.alltrue(diff < 0):
        # Resetting row
        # Pick the row with the smallest points automatically
        row = self.table_points.sum(-1).argmin()
        col = 0
        if self.debug:
          print("Player %i card: %i resetting row %i" % (player, card, row))
        self._reset_row(row, player)
      else:
        # Appending row
        diff_argsort = diff.argsort()
        diff.sort()
        row = diff_argsort[diff > 0][0]
        col = self.table[row].argmax() + 1
        if col >= 5:
          if self.debug:
            print("Player %i card: %i, appending to row: %i and take 5" % (player, card, row))
          self._reset_row(row, player)
          col = 0
        elif self.debug:
          print("Player %i card: %i, appending to row: %i, col: %i" % (player, card, row, col))
      self.table[row, col] = card
      self.table_points = np.take(self.points, self.table)
    self.round += 1
    if self.round == self.max_hand:
      done = True
    else:
      done = False
    observation = self.table, self.hands[0]
    return observation, -self.penalties[0], done, {}

  def reset(self):
    self.deck = np.arange(1, self.max_card + 1, dtype=int)
    np.random.shuffle(self.deck)
    self.hands = self._draw_card((self.sides, self.max_hand))
    self.table = np.zeros((self.table_rows, 5), dtype=int)
    self.table[:, 0] = self._draw_card(self.table_rows)
    self.table_points = np.take(self.points, self.table)
    self.accum_penalties = np.zeros(self.sides)
    observation = self.table, self.hands[0]
    return observation

  def render(self, mode='human', close=False):
    print("Table:")
    print(self.table)
    print("Hands:")
    print(self.hands)
    print("Accumulative panelty")
    print(self.accum_penalties)
