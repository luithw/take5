import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


DEBUG = False
ONE_HOT = True


class Take5Env(MultiAgentEnv):
  metadata = {'render.modes': ['human']}

  def __init__(self, config):
    self.sides = config.get("sides", 3)
    self.illegal_moves_limit = config.get("illegal_moves_limit", 3) * self.sides

    self.players = ["p_" + str(i) for i in range(self.sides)]
    self.largest_card = 104
    self.max_hand = 10
    self.n_rows = 4
    self.n_table_cards = self.n_rows * 5
    self.points = np.ones(self.largest_card+1)
    self.points[0] = 0
    self.round = 0
    for c in range(5, self.largest_card, 5):
        self.points[c] = 2
    for c in range(10, self.largest_card, 10):
        self.points[c] = 3
    for c in range(11, self.largest_card, 11):
        self.points[c] = 5

    self.reward_range = (0, 55)
    self.observation_space = spaces.Box(low=0, high=self.largest_card, shape=(self.n_table_cards + self.max_hand, self.largest_card + 1), dtype=np.int)
    self.action_space = spaces.Discrete(10)
    self.illegal_moves_count = 0


  def _draw_card(self, shape):
    drawn = self.deck[:np.prod(shape)].reshape(shape)
    self.deck = self.deck[np.prod(shape):]
    return drawn

  def _reset_row(self, row, player):
    self.penalties[player] += self.table_points.sum(-1)[row]
    self.accum_penalties[player] += self.table_points.sum(-1)[row]
    self.table[row] = 0

  def step(self, action_dict):
    # Check if the action taken has a valid card, otherwise return 0 reward
    for player, hand in zip(self.players, self.hands):
      availability = (hand > 0).astype(float)
      action = action_dict[player]
      if availability[action] == 0:
        if self.illegal_moves_count < self.illegal_moves_limit:
          self.illegal_moves_count += 1
          done = False
        else:
          done = True
        return self._get_obs(), 0, done, {"legal_move": self.illegal_moves_count}
      else:
        self.illegal_moves_count = 0


    played_cards = []
    self.player_played_card = np.zeros(self.sides)
    for i, (player, hand) in enumerate(zip(self.players, self.hands)):
      action = action_dict[player]
      played_card = hand[action]
      played_cards.append((played_card, i))
      self.player_played_card[i] = played_card
      hand[action] = 0
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
        if DEBUG:
          print("Player %i card: %i resetting row %i" % (player, card, row))
        self._reset_row(row, player)
      else:
        # Appending row
        diff_argsort = diff.argsort()
        diff.sort()
        row = diff_argsort[diff > 0][0]
        col = self.table[row].argmax() + 1
        if col >= 5:
          if DEBUG:
            print("Player %i card: %i, appending to row: %i and take 5" % (player, card, row))
          self._reset_row(row, player)
          col = 0
        elif DEBUG:
          print("Player %i card: %i, appending to row: %i, col: %i" % (player, card, row, col))
      self.table[row, col] = card
      self.table_points = np.take(self.points, self.table)
    self.round += 1
    if self.round == self.max_hand:
      done = True
    else:
      done = False
    reward = 0
    for i, penalty in enumerate(self.penalties):
      if i == 0:
        reward -= penalty
      else:
        reward += penalty
    if DEBUG:
      print("Player reward: %i" % reward)

    return self._get_obs(), reward, done, {"legal_move": True}

  def _get_obs(self):
    all_obs = {}
    for player, hand in zip(self.players, self.hands):
      obs = np.concatenate((self.table.flatten(), hand))
      obs_one_hot = np.zeros([obs.size, self.largest_card + 1])
      obs_one_hot[np.arange(obs.size), obs] = 1
      self.table_points = np.take(self.points, self.table)
      obs_one_hot[np.arange(self.n_table_cards), obs[:self.n_table_cards]] = self.table_points.flatten()
      all_obs[player] = obs_one_hot
    return all_obs

  def reset(self):
    self.deck = np.arange(1, self.largest_card + 1, dtype=int)
    np.random.shuffle(self.deck)
    self.hands = self._draw_card((self.sides, self.max_hand))
    self.table = np.zeros((self.n_rows, 5), dtype=int)
    self.table[:, 0] = self._draw_card(self.n_rows)
    self.table_points = np.take(self.points, self.table)
    self.accum_penalties = np.zeros(self.sides)
    observation = {"table": self.table, "hand": self.hands[0]}
    self.round = 0
    self.player_played_card = np.zeros(self.sides)
    self.illegal_moves_count = 0
    return self._get_obs()

  def render(self, mode='human', close=False):
    print("=====Round %i======" % self.round)
    print("Table:")
    print(self.table)
    if self.illegal_moves_count:
      print("Player 0 played illegal card %i times." % self.illegal_moves_count)
    else:
      for player, (played, hand, penalty) in enumerate(zip(self.player_played_card, self.hands, self.accum_penalties)):
          print("%i played %i, remaining cards: %r, penalty: %i" % (player, played, [c for c in hand if c], penalty))
