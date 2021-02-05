import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class Take5Env(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, sides=2):
    self.sides = sides
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

  def step(self, action):
    played_cards = []
    for h, hand in enumerate(self.hands):
      if h==0:
        a = action
      else:
        a = hand.argmax()
      played_cards.append((hand[action], h))
      hand[action] = 0
    played_cards = np.array(played_cards, dtype=[('card', int), ('player', int)])
    played_cards = np.sort(played_cards, order='card')
    self.penalties = np.zeros(self.sides)

    for card, player in played_cards:
      diff = card - self.table.max(-1)
      if np.alltrue(diff < 0):
        # Resetting row
        row = self.table_points.sum(-1).argmin()
        self.penalties[player] -= self.table_points.sum(-1).min()
        self.table[row] = 0
      else:
        # Appending row
        row = diff[diff > 0].argmin()
        col = self.table[row].argmin()
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
    self.penalties = np.zeros(self.sides)
    observation = self.table, self.hands[0]
    return observation

  def render(self, mode='human', close=False):
    print("self.table: %r" % self.table)
    print("self.hands: %r" % self.hands)
