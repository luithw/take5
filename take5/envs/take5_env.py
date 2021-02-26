import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

DEBUG = False


class Take5Env(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_SIDES = 7
    MIN_SIDES = 2
    LARGEST_CARD = 104
    MAX_HAND = 10
    N_ROWS = 4
    N_COLUMNS = 5
    PENALTIES = [2, 3, 5]  # must be in ascending order

    def __init__(self, config):
        self.N_CARDS_TABLE = self.N_ROWS * self.N_COLUMNS

        self.multi_agent = config.get("multi_agent", False)
        self.sides = config.get("sides", 3)
        if self.sides < self.MIN_SIDES or self.sides > self.MAX_SIDES:
            raise ValueError(f"Number of sides is outside range of {self.MIN_SIDES} - {self.MAX_SIDES}")

        self.players = ["p_" + str(i) for i in range(self.sides)]
        self.round = 0

        # Set the penalty associated with each card
        self.card_penalties = np.ones(self.LARGEST_CARD + 1)
        self.card_penalties[0] = 0
        for c in range(5, self.LARGEST_CARD, 5):
            self.card_penalties[c] = self.PENALTIES[0]
        for c in range(10, self.LARGEST_CARD, 10):
            self.card_penalties[c] = self.PENALTIES[1]
        for c in range(11, self.LARGEST_CARD, 11):
            self.card_penalties[c] = self.PENALTIES[2]

        # Max positive reward when all other players have to take 5 cards all with a penalty of 5
        # Max negative reward when only the current player has to take 5 cards all with a penalty of 5
        self.reward_range = (-25.0, 25.0)

        # +2 to include the timer and number of sides in the state
        self.observation_rows = self.N_CARDS_TABLE + self.MAX_HAND + 2
        # (highlighted, has_card, card_value, mod5, mod10, mod11)
        self.observation_columns = 6
        self.observation_shape = (self.observation_rows, self.observation_columns)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.observation_shape, dtype=np.float32)

        self.action_space = spaces.Discrete(3)

    def _draw_card(self, shape):
        n_cards = np.prod(shape)
        drawn = self.deck[:n_cards].reshape(shape)
        self.deck = self.deck[n_cards:]
        return drawn

    def _take_row(self, row, player_ix):
        table_card_penalties = np.take(self.card_penalties, self.table)
        penalty = table_card_penalties[row].sum()
        self.penalties[player_ix] += penalty
        self.total_penalties[player_ix] += penalty
        self.table[row] = 0

    def _parse_action(self, actions, player_ix):
        if self.multi_agent:
            action = actions[self.players[player_ix]]
        elif player_ix == 0:
            action = actions
        else:
            hand = self.hands[player_ix]
            hand_hl = self.hand_hl[player_ix]
            # Use max card policy for all non-agents
            card_idx = np.argmax(hand_hl)
            to_select_card_idx = np.argmax(hand)
            jumps = to_select_card_idx - card_idx
            if jumps < 0:
                action = 0
            elif jumps > 0:
                action = 1
            else:
                # current card is the highest so select it
                action = 2

        return action

    def get_action_meanings(self):
        return ['LEFT', 'RIGHT', 'SKIP']

    def _move_highlight_card(self, player_ix, jump_direction):
        hand = self.hands[player_ix]
        hand_hl = self.hand_hl[player_ix]

        jumps = jump_direction
        card_idx = np.argmax(hand_hl)
        while hand[(card_idx + jumps) % self.MAX_HAND] == 0:
            jumps += jump_direction

        self.hand_hl[player_ix] = np.roll(hand_hl, jumps)

    def step(self, actions):
        if self.reset_played_cards:
            self.player_played_card = np.zeros(self.sides, dtype=int)
            self.reset_played_cards = False

        self.moves_left -= 1

        # Card selection controlls
        for player_ix, player_id in enumerate(self.players):
            action = self._parse_action(actions, player_ix)
            # Move highlight left with wrap around and skip already played cards
            if action == 0:
                self._move_highlight_card(player_ix, -1)
            # Move highlight right with wrap around and skip already played cards
            elif action == 1:
                self._move_highlight_card(player_ix, 1)
            elif action == 2:
                # Don't do anything on SKIP action
                pass

        # When the timer runs out, play the round
        if self.moves_left <= 0:
            cards_left = self.MAX_HAND - self.round - 1
            self.moves_left = cards_left // 2

            # Play the currently highlighted card
            for player_ix, player_id in enumerate(self.players):
                hand = self.hands[player_ix]
                self.player_played_card[player_ix] = hand[self.hand_hl[player_ix]]
                hand[self.hand_hl[player_ix]] = 0  # remove card from hand
                if cards_left > 0:
                    jumps = cards_left // 2
                    index = 0
                    while jumps > 0 or hand[index] == 0:
                        if hand[index] != 0:
                            jumps -= 1
                        index += 1
                    self.hand_hl[player_ix] = np.zeros(self.MAX_HAND, dtype=bool)
                    self.hand_hl[player_ix, index] = True

            self.penalties = np.zeros(self.sides)

            # Match player_ix with played card
            player_with_card = np.array(list(enumerate(self.player_played_card)), dtype=[
                ('player', int), ('card', int)])
            # Sort played cards
            player_with_card = np.sort(player_with_card, order="card")

            for player_ix, card in player_with_card:
                diff = card - self.table.max(-1)
                if np.alltrue(diff < 0):
                    # Resetting row if the played card is smaller than all rows
                    # Pick the row with the smallest penalty automatically
                    table_card_penalties = np.take(self.card_penalties, self.table)
                    row = table_card_penalties.sum(-1).argmin()
                    col = 0
                    if DEBUG:
                        print("Player %i card: %i resetting row %i" %
                              (player_ix, card, row))
                    self._take_row(row, player_ix)
                else:
                    # Appending card to the row with closest value
                    diff_argsort = diff.argsort()
                    diff.sort()
                    row = diff_argsort[diff > 0][0]
                    col = self.table[row].argmax() + 1
                    if col >= 5:
                        if DEBUG:
                            print("Player %i card: %i, appending to row: %i and take 5" % (
                                player_ix, card, row))
                        self._take_row(row, player_ix)
                        col = 0
                    elif DEBUG:
                        print("Player %i card: %i, appending to row: %i, col: %i" %
                              (player_ix, card, row, col))
                # Place the player's card on the table
                self.table[row, col] = card
            self.round += 1
            self.reset_played_cards = True

        done = self.round == self.MAX_HAND

        return self._get_obs(), self._get_rewards(), self._get_dones(done), self._get_infos({})

    # def _get_obs(self):
    #     all_obs = {}
    #     for player, hand in zip(self.players, self.hands):
    #         obs = np.concatenate((self.table.flatten(), hand))
    #         obs_one_hot = np.zeros([obs.size, self.largest_card + 1])
    #         obs_one_hot[np.arange(obs.size), obs] = 1
    #         card_points = np.take(self.card_points, obs)
    #         obs_one_hot[np.arange(len(card_points)), obs] = card_points
    #         all_obs[player] = obs_one_hot
    #         if not self.multi_agent:
    #             return obs_one_hot
    #     return all_obs

    def _get_obs(self):
        all_obs = {}
        for player_ix, player_id in enumerate(self.players):
            cards = np.concatenate(
                (self.table.flatten(), self.hands[player_ix]))
            cards_encoded = self._encode_cards(cards)

            table_hl = np.repeat(self.table_row_hl[player_ix], self.N_COLUMNS)
            highlight = np.concatenate((table_hl, self.hand_hl[player_ix]))
            highlight = highlight.astype(np.float32)

            timer_one_hot = np.zeros(self.observation_columns, dtype=np.float32)
            timer_one_hot[self.moves_left] = 1.0

            sides_one_hot = np.zeros(self.observation_columns, dtype=np.float32)
            sides_one_hot[self.sides - 1] = 1.0

            obs = np.hstack((np.expand_dims(highlight, -1), cards_encoded))
            obs = np.vstack((obs, timer_one_hot))
            obs = np.vstack((obs, sides_one_hot))

            if not self.multi_agent:
                return obs
            all_obs[player_id] = obs
        return all_obs

    def _encode_cards(self, cards):
        state = np.zeros((len(cards), 5), dtype=np.float32)
        state[:, 0] = np.array(cards != 0, dtype=np.float32)
        state[:, 1] = cards.astype(np.float32) / self.LARGEST_CARD
        penalties = np.take(self.card_penalties, cards)
        state[:, 2] = (penalties == self.PENALTIES[0]).astype(np.float32)
        state[:, 3] = (penalties == self.PENALTIES[1]).astype(np.float32)
        state[:, 4] = (penalties == self.PENALTIES[2]).astype(np.float32)
        return state

    # def _get_binary_encoding(self, values, base=None):
    #     if base is None:
    #         base = self.binary_values_per_card
    #
    #     pows_of_two = 1 << np.arange(base)
    #     return ((values[:, None] & pows_of_two) > 0).astype(np.float32)

    def _get_rewards(self):
        # Zero-sum rewards
        float_penalties = -self.penalties.astype(np.float32)
        rewards = float_penalties - float_penalties.mean()

        if DEBUG:
            print("Player rewards: %r" % rewards)
        if not self.multi_agent:
            return rewards[0]

        rewards = {player_id: reward for player_id, reward in zip(self.players, rewards)}
        return rewards

    def _get_dones(self, done):
        if self.multi_agent:
            dones = {player_id: done for player_id in self.players}
            dones["__all__"] = done
            return dones

        return done

    def _get_infos(self, info):
        if self.multi_agent:
            return {player_id: info for player_id in self.players}
        return info

    def reset(self):
        self.round = 0

        self.deck = np.arange(1, self.LARGEST_CARD + 1, dtype=int)
        np.random.shuffle(self.deck)

        self.hands = self._draw_card((self.sides, self.MAX_HAND))
        self.table = np.zeros((self.N_ROWS, self.N_COLUMNS), dtype=int)
        self.table[:, 0] = self._draw_card(self.N_ROWS)

        self.table_row_hl = np.zeros((self.sides, self.N_ROWS), dtype=bool)
        self.hand_hl = np.zeros((self.sides, self.MAX_HAND), dtype=bool)
        # initially highlight the middle card in hand
        self.hand_hl[:, self.MAX_HAND // 2] = True
        self.moves_left = self.MAX_HAND // 2

        self.penalties = np.zeros(self.sides, dtype=int)
        self.total_penalties = np.zeros(self.sides, dtype=int)
        self.player_played_card = np.zeros(self.sides, dtype=int)
        self.reset_played_cards = False

        return self._get_obs()

    def render(self, mode='human', close=False):
        print("=====Round %i======" % self.round)
        print("Table:")
        print(self.table)
        for player_ix in range(self.sides):
            played = self.player_played_card[player_ix]
            hand = self.hands[player_ix]
            penalty = self.total_penalties[player_ix]
            highlighted = self.hands[player_ix, self.hand_hl[player_ix]]
            print("%i played %i, remaining cards: %r, highlighted: %i, penalty: %i" %
                  (player_ix, played, [c for c in hand if c], highlighted, penalty))
