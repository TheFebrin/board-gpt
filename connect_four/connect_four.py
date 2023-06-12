import numpy as np
from abc import ABC, abstractmethod
import time


class IllegalMove(Exception):
    def __init__(self, msg: str, player_id: str):
        self.value = msg
        self.player_id = player_id
   
    def __str__(self):
        return(repr(self.value))


class ConnectFour:
    """
    First player controls pieces equal to 1, second player controls pieces equal to 2.
    Empty cells have value 0.
    """
    def __init__(self, rows_count: int = 6, columns_count: int = 7):
        self.empty_cell = 0
        self.board = np.array([
            [self.empty_cell for i in range(columns_count)]
            for i in range(rows_count)
        ])
        self.columns_count = columns_count
        self.rows_count = rows_count

    def print_board(self):
        for row in self.board:
            print(row)

    def generate_legal_moves(self) -> list[int]:
        res = [
            column_idx for column_idx in range(self.columns_count)
            if self.is_move_possible(column_idx)
        ]
        return res

    def make_move(self, piece: int, column: int):
        if not self.is_move_possible(column):
            raise IllegalMove(f"Illegal move: {column}", piece)
        self.apply_move(piece, column)

    def is_move_possible(self, column: int) -> bool:
        return self.board[0][column] == self.empty_cell

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
                    self.board[r][c] == self.board[r][c+1] ==
                    self.board[r][c+2] == self.board[r][c+3] != self.empty_cell
                ):
                    return self.board[r][c]

        # Vertical position
        for c in range(self.columns_count):
            for r in range(self.rows_count - 3):
                if (
                    self.board[r][c] == self.board[r+1][c] ==
                    self.board[r+2][c] == self.board[r+3][c] != self.empty_cell
                ):
                    return self.board[r][c]

        # Diagonal slope down
        for c in range(self.columns_count - 3):
            for r in range(self.rows_count - 3):
                if (
                    self.board[r][c] == self.board[r+1][c+1] ==
                    self.board[r+2][c+2] == self.board[r+3][c+3] != self.empty_cell
                ):
                    return self.board[r][c]

        # Diagonal slope up
        for c in range(self.columns_count - 3):
            for r in range(3, self.rows_count):
                if (
                    self.board[r][c] == self.board[r-1][c+1] ==
                    self.board[r-2][c+2] == self.board[r-3][c+3] != self.empty_cell
                ):
                    return self.board[r][c]
        return -1


class Agent(ABC):
    @abstractmethod
    def choose_move(self) -> int:
        raise NotImplementedError


class RandomAgent(ABC):
    def __init__(self, game: ConnectFour):
        self.game = game

    def choose_move(self) -> int:
        possible_moves = self.game.generate_legal_moves()
        selected_move = np.random.choice(possible_moves)
        return selected_move


class Arena:
    def __init__(
            self,
            agent_one: Agent,
            agent_two: Agent,
            game: ConnectFour
    ):
        self.players = {
            1: agent_one,
            2: agent_two
        }
        self.game = game
        self.transcript = []

    def play(self) -> int:
        while True:
            for players_piece, agent in self.players.items():
                move = agent.choose_move()
                self.game.make_move(players_piece, move)
                self.transcript.append(move)
                winner = self.game.is_game_finished()
                if winner != -1:
                    return winner
                

def generate_random_game() -> list[int]:
    cf = ConnectFour()
    agentone = RandomAgent(cf)
    agenttwo = RandomAgent(cf)
    arena = Arena(agentone, agenttwo, cf)
    return arena.transcript
        

# cf = ConnectFour()
# agentone = RandomAgent(cf)
# agenttwo = RandomAgent(cf)
# arena = Arena(agentone, agenttwo, cf)
# winner = arena.play()
# print("Game state: {}".format(winner))
# arena.game.print_board()
# print(arena.transcript)
# cf.board[5][1] = 1
# cf.board[5][2] = 2
# cf.board[4][1] = 1
# cf.board[0][1] = 3
# cf.print_board()
# print("MOVES:", cf.generate_legal_moves())
# print(np.random.choice(cf.generate_legal_moves()))
# print()
# print(cf.board[:, 1])
# print(cf.board[0][1])
# print(cf.board[0,:])
# print(cf.board.shape)
