from abc import ABC, abstractmethod
import numpy as np
from typing import List
from connect_four.connect_four import ConnectFour
from mingpt.utils import sample
from mingpt.model import GPT
import torch
import random
from connect_four.data_processing_config import DatasetPreprocessingConfig


class Agent(ABC):
    @abstractmethod
    def choose_move(self) -> int:
        raise NotImplementedError

    def set_game(self, game: ConnectFour) -> None:
        self._game = game

    def set_player_id(self, player_id: int) -> None:
        assert player_id in [1, 2]
        self._player_id = player_id


class RandomAgent(Agent):
    def __init__(self, game: ConnectFour, name: str = ""):
        self._game = game
        self.name = name

    def choose_move(self) -> int:
        possible_moves = self._game.generate_legal_moves()
        selected_move = np.random.choice(possible_moves)
        return selected_move


class MinMaxAgent(Agent):
    def __init__(
        self, game: ConnectFour, max_depth: int, player_id: int, name: str = ""
    ):
        self._game = game
        self._max_depth = max_depth
        self.set_player_id(player_id)
        self.name = name

    def __str__(self) -> str:
        return "Min Max Agent"

    def choose_move(self) -> int:
        best_score = -np.inf
        best_move = None
        for move in self._game.generate_legal_moves(shuffle=True):
            self._game.make_move(piece=self._player_id, column=move)
            score = self._minimax(
                depth=0, is_maximizing=False, alpha=-np.inf, beta=np.inf
            )
            self._game.undo_move(column=move)
            # print(f"[{self._player_id}] Move: ", move, "Score: ", score)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(self, depth: int, is_maximizing: bool, alpha: int, beta: int) -> int:
        if self._game.is_game_finished() != -1 or depth >= self._max_depth:
            return self.evaluate_board()

        if is_maximizing:
            max_eval = -np.inf
            for move in self._game.generate_legal_moves(shuffle=True):
                self._game.make_move(piece=self._player_id, column=move)
                eval = self._minimax(
                    depth=depth + 1, is_maximizing=False, alpha=alpha, beta=beta
                )
                self._game.undo_move(column=move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = np.inf
            for move in self._game.generate_legal_moves(shuffle=True):
                self._game.make_move(
                    piece=2 if self._player_id == 1 else 1, column=move
                )
                eval = self._minimax(
                    depth=depth + 1, is_maximizing=True, alpha=alpha, beta=beta
                )
                self._game.undo_move(column=move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def undo_move(self, move: int) -> None:
        # This method should remove the last piece played in the specified column.
        self._game.undo_move(move)

    def evaluate_board(self) -> int:
        """
        Scores the board based on the player's pieces' strategic positioning.
            it gives preference to the center column,
            as control of the center is generally advantageous in Connect Four.
            It also checks all possible four-piece combinations (windows)
            in the horizontal, vertical, and diagonal directions.
        """
        score: int = 0

        # Center column preference
        center_array: List[int] = [
            int(i) for i in list(self._game.board[:, self._game.columns_count // 2])
        ]
        center_count = center_array.count(self._player_id)

        score += center_count * 3

        # Check horizontal locations for score
        for r in range(self._game.rows_count):
            row_array = [int(i) for i in list(self._game.board[r, :])]
            for c in range(self._game.columns_count - 3):
                window = row_array[c : c + 4]
                score += self.evaluate_window(window)

        # Check vertical locations for score
        for c in range(self._game.columns_count):
            col_array = [int(i) for i in list(self._game.board[:, c])]
            for r in range(self._game.rows_count - 3):
                window = col_array[r : r + 4]
                score += self.evaluate_window(window)

        # Check positively sloped diagonals
        for r in range(self._game.rows_count - 3):
            for c in range(self._game.columns_count - 3):
                window = [self._game.board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window)

        # Check negatively sloped diagonals
        for r in range(self._game.rows_count - 3):
            for c in range(self._game.columns_count - 3):
                window = [self._game.board[r + 3 - i][c + i] for i in range(4)]
                score += self.evaluate_window(window)

        return score

    def evaluate_window(self, window: List[int]) -> int:
        score = 0
        opp_piece = 2 if self._player_id == 1 else 1
        my_window_count = window.count(self._player_id)
        opp_window_count = window.count(opp_piece)

        if my_window_count == 4:
            score += 100
        elif my_window_count == 3 and window.count(self._game.empty_cell) == 1:
            score += 5
        elif my_window_count == 2 and window.count(self._game.empty_cell) == 2:
            score += 2

        if opp_window_count == 3 and window.count(self._game.empty_cell) == 1:
            score -= 80  # increased penalty for opponent about to win

        return score


class GPTAgent(Agent):
    def __init__(
        self,
        model: GPT,
        game: ConnectFour,
        preprocessing_config: DatasetPreprocessingConfig,
        device: int,
        first_move: int = None,
        name: str = "",
        randomness: float = 0.0,
    ):
        self.model = model
        self.device = device
        self._game = game
        self.config = preprocessing_config
        if first_move is None:
            first_move = np.random.choice(list(range(game.columns_count)))
        self.first_move = first_move
        self.name = name
        self.randomness = randomness

    def __str__(self) -> str:
        return "GPT Agent"
    
    def choose_move(self) -> int:
        if random.uniform(0, 1.0) <= self.randomness:
            possible_moves = self._game.generate_legal_moves()
            selected_move = np.random.choice(possible_moves)
            return selected_move
        elif len(self._game.history) == 0:
            return self.first_move
        else:
            x = torch.tensor(
                [self.config.to_model_repr[s] for s in self._game.history],
                dtype=torch.long,
            )[None, ...].to(self.device)
            y = sample(self.model, x, 1, temperature=1.0)[0]
            completion = [self.config.from_model_repr[int(i)] for i in y if i != -1]
            return completion[-1]
