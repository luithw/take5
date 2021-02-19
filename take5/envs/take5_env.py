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
        self.illegal_moves_limit = config.get(
            "illegal_moves_limit", 3) * self.sides

        self.players = ["p_" + str(i) for i in range(self.sides)]
        self.largest_card = 104
        self.max_hand = 10
        self.n_rows = 4
        self.n_columns = 5
        self.n_table_cards = self.n_rows * self.n_columns
        self.round = 0
        self.binary_values_per_card = int(np.ceil(np.log2(self.largest_card)))

        self.points = np.ones(self.largest_card+1)
        self.points[0] = 0
        for c in range(5, self.largest_card, 5):
            self.points[c] = 2
        for c in range(10, self.largest_card, 10):
            self.points[c] = 3
        for c in range(11, self.largest_card, 11):
            self.points[c] = 5

        # max positive reward when other player has to take 5 cards all with a penalty of 5
        # max negative reward when current player has to take 5 cards all with a penalty of 5
        self.reward_range = (-25.0, 25.0)

        observation_width = self.n_table_cards + self.max_hand + 1  # +1 for the timer
        # 7 numbers to binary encode the card and 1 boolean for highlighted
        observation_height = self.binary_values_per_card + 1
        self.observation_shape = (observation_width, observation_height)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.observation_shape, dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        # self.illegal_moves_count = 0

    def _draw_card(self, shape):
        drawn = self.deck[:np.prod(shape)].reshape(shape)
        self.deck = self.deck[np.prod(shape):]
        return drawn

    def _reset_row(self, row, player_ix):
        self.penalties[player_ix] += self.table_points.sum(-1)[row]
        self.total_penalties[player_ix] += self.table_points.sum(-1)[row]
        self.table[row] = 0

    def get_action_meanings(self):
        return ['LEFT', 'RIGHT', 'PLAY']

    def step(self, action_dict):
        self.penalties = np.zeros(self.sides)

        # Card selection controlls
        for player_ix, player_id in enumerate(self.players):
            if self.player_played_card[player_ix] != 0:
                continue

            hand = self.hands[player_ix]
            hand_hl = self.hand_hl[player_ix]
            action = action_dict[player_id]

            # Move highlight left with wrap around and skip already played cards
            if action == 0:
                jumps = -1
                card_idx = np.argmax(hand_hl)
                while hand[(card_idx + jumps) % self.max_hand] == 0:
                    jumps -= 1
                self.hand_hl[player_ix] = np.roll(hand_hl, jumps)

            # Move highlight right with wrap around and skip already played cards
            elif action == 1:
                jumps = 1
                card_idx = np.argmax(hand_hl)
                while hand[(card_idx + jumps) % self.max_hand] == 0:
                    jumps += 1
                self.hand_hl[player_ix] = np.roll(hand_hl, jumps)

            # Play the currently highlighted card
            elif action == 2:
                self.player_played_card[player_ix] = hand[self.hand_hl[player_ix]]

        # Check if all players have selected a card, then play the round
        if self.player_played_card.min() != 0:
            # Match player_ix with played card
            player_with_card = np.array(list(enumerate(self.player_played_card)), dtype=[
                ('player', int), ('card', int)])
            # Sort played cards
            player_with_card = np.sort(player_with_card, order="card")

            for player_ix, card in player_with_card:
                diff = card - self.table.max(-1)
                if np.alltrue(diff < 0):
                    # Resetting row
                    # Pick the row with the smallest points automatically
                    row = self.table_points.sum(-1).argmin()
                    col = 0
                    if DEBUG:
                        print("Player %i card: %i resetting row %i" %
                            (player_ix, card, row))
                    self._reset_row(row, player_ix)
                else:
                    # Appending row
                    diff_argsort = diff.argsort()
                    diff.sort()
                    row = diff_argsort[diff > 0][0]
                    col = self.table[row].argmax() + 1
                    if col >= 5:
                        if DEBUG:
                            print("Player %i card: %i, appending to row: %i and take 5" % (
                                player_ix, card, row))
                        self._reset_row(row, player_ix)
                        col = 0
                    elif DEBUG:
                        print("Player %i card: %i, appending to row: %i, col: %i" %
                            (player_ix, card, row, col))
                self.table[row, col] = card
                self.table_points = np.take(self.points, self.table)
            self.round += 1
            if self.round == self.max_hand:
                done = True
            else:
                done = False

        return self._get_obs(), self._get_rewards(), self._get_dones(done), self._get_infos({})

    def _get_obs(self):
        all_obs = {}
        for player_ix, player_id in enumerate(self.players):
            has_picked_card = self.player_played_card[player_ix] != 0
            # TODO: when implementing custom row selection this causes a bug
            if has_picked_card:
                all_obs[player_id] = None
                continue

            cards = np.concatenate(
                (self.table.flatten(), self.hands[player_ix]))
            cards_binary = self._get_binary_encoding(cards)

            table_hl = np.repeat(self.table_row_hl[player_ix], self.n_columns)
            highlight = np.concatenate((table_hl, self.hand_hl[player_ix]))
            highlight = highlight.astype(np.float32)

            timer_binary = self._get_binary_encoding(
                self.moves_left[player_ix], self.observation_shape[1])

            obs = np.hstack((np.expand_dims(highlight, -1), cards_binary))
            obs = np.vstack((obs, timer_binary))

            all_obs[player_id] = obs
        return all_obs

    def _get_binary_encoding(self, values, base=None):
        if base is None:
            base = self.binary_values_per_card

        pows_of_two = 1 << np.arange(base)
        return ((values[:, None] & pows_of_two) > 0).astype(np.float32)

    def _get_rewards(self):
        # Zero-sum rewards
        rewards = self.penalties - self.penalties.sum()
        rewards = {player_id: reward for player_id,
                   reward in zip(self.players, self.penalties)}
        if DEBUG:
            print("Player rewards: %r" % rewards)
        return rewards

    def _get_dones(self, done):
        dones = {player_id: done for player_id in self.players}
        dones["__all__"] = done
        return dones

    def _get_infos(self, info):
        return {player_id: info for player_id in self.players}

    def reset(self):
        self.round = 0

        self.deck = np.arange(1, self.largest_card + 1, dtype=int)
        np.random.shuffle(self.deck)

        self.hands = self._draw_card((self.sides, self.max_hand))
        self.table = np.zeros((self.n_rows, self.n_columns), dtype=int)
        self.table[:, 0] = self._draw_card(self.n_rows)

        self.table_row_hl = np.zeros((self.sides, self.n_rows), dtype=bool)
        self.hand_hl = np.zeros((self.sides, self.max_hand), dtype=bool)
        # initially highlight the middle card in hand
        self.hand_hl[:, self.max_hand // 2] = True
        self.moves_left = np.zeros(self.sides)
        self.moves_left += self.max_hand // 2

        self.table_points = np.take(self.points, self.table)
        self.total_penalties = np.zeros(self.sides)

        self.player_played_card = np.zeros(self.sides)
        # self.illegal_moves_count = 0
        return self._get_obs()

    def render(self, mode='human', close=False):
        print("=====Round %i======" % self.round)
        print("Table:")
        print(self.table)
        # if self.illegal_moves_count:
        #     print("Player 0 played illegal card %i times." %
        #           self.illegal_moves_count)
        # else:
        for player, (played, hand, penalty) in enumerate(zip(self.player_played_card, self.hands, self.total_penalties)):
            print("%i played %i, remaining cards: %r, penalty: %i" %
                  (player, played, [c for c in hand if c], penalty))
