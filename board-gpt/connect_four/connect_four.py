import numpy as np
import random
from typing import *


class IllegalMove(Exception):
    def __init__(self, msg: str, player_id: str):
        self.value = msg
        self.player_id = player_id

    def __str__(self):
        return repr(self.value)


class ConnectFour:
    """
    First player controls pieces equal to 1, second player controls pieces equal to 2.
    Empty cells have value 0.
    """

    def __init__(self, rows_count: int = 6, columns_count: int = 7):
        self.empty_cell = 0
        self.board = np.array(
            [[self.empty_cell for i in range(columns_count)] for i in range(rows_count)]
        )
        self.columns_count = columns_count
        self.rows_count = rows_count
        self.history = []

    def print_board(self):
        for row in self.board:
            print(row)

    def generate_legal_moves(self, shuffle=False) -> List[int]:
        res = [
            column_idx
            for column_idx in range(self.columns_count)
            if self.is_move_possible(column_idx)
        ]
        if shuffle:
            random.shuffle(res)
        return res

    def make_move(self, piece: int, column: int) -> None:
        if not self.is_move_possible(column):
            raise IllegalMove(f"Illegal move: {column}", piece)
        self.apply_move(piece, column)
        self.history.append(column)

    def undo_move(self, column: int) -> None:
        """
        Removes the last piece played in the given column.
        Assumes that the given column is not empty.
        """
        for row_idx in range(len(self.board[:, column])):
            if self.board[row_idx][column] != self.empty_cell:
                self.board[row_idx][column] = self.empty_cell
                return

    def is_move_possible(self, column: int) -> bool:
        return (
            0 <= column < self.columns_count
            and self.board[0][column] == self.empty_cell
        )

    def apply_move(self, piece: int, selected_column: int) -> None:
        """
        Assumes that a given move is legal.
        """
        for row_idx in range(len(self.board[:, selected_column])):
            if self.board[row_idx][selected_column] == self.empty_cell:
                continue
            else:
                self.board[row_idx - 1][selected_column] = piece
                return
        self.board[-1][selected_column] = piece

    def is_game_finished(self) -> int:
        """
        Returns
            int:
                -1, if the game is not finished.
                0, if it's a draw
                otherwise: piece of a player who won
        """
        if len(self.generate_legal_moves()) == 0:
            return 0
        # Horizontal position
        for c in range(self.columns_count - 3):
            for r in range(self.rows_count):
                if (
                    self.board[r][c]
                    == self.board[r][c + 1]
                    == self.board[r][c + 2]
                    == self.board[r][c + 3]
                    != self.empty_cell
                ):
                    return self.board[r][c]

        # Vertical position
        for c in range(self.columns_count):
            for r in range(self.rows_count - 3):
                if (
                    self.board[r][c]
                    == self.board[r + 1][c]
                    == self.board[r + 2][c]
                    == self.board[r + 3][c]
                    != self.empty_cell
                ):
                    return self.board[r][c]

        # Diagonal slope down
        for c in range(self.columns_count - 3):
            for r in range(self.rows_count - 3):
                if (
                    self.board[r][c]
                    == self.board[r + 1][c + 1]
                    == self.board[r + 2][c + 2]
                    == self.board[r + 3][c + 3]
                    != self.empty_cell
                ):
                    return self.board[r][c]

        # Diagonal slope up
        for c in range(self.columns_count - 3):
            for r in range(3, self.rows_count):
                if (
                    self.board[r][c]
                    == self.board[r - 1][c + 1]
                    == self.board[r - 2][c + 2]
                    == self.board[r - 3][c + 3]
                    != self.empty_cell
                ):
                    return self.board[r][c]
        return -1
