from abc import ABC, abstractmethod
import time
from tqdm import tqdm
from typing import Tuple
from connect_four.agents import Agent, RandomAgent, MinMaxAgent
from connect_four.connect_four import ConnectFour


class Arena:
    def __init__(self, agent_one: Agent, agent_two: Agent, game: ConnectFour):
        self.players = {1: agent_one, 2: agent_two}
        self.game = game

    def play(self) -> int:
        while True:
            for players_piece, agent in self.players.items():
                move = agent.choose_move()
                self.game.make_move(players_piece, move)

                # print(f"[{players_piece}] Move: ", move)
                # self.game.print_board()
                # print("\n====================\n")

                winner = self.game.is_game_finished()
                if winner != -1:
                    return winner


def generate_random_game(cf: ConnectFour) -> Tuple[int, ConnectFour]:
    # agent_one = MinMaxAgent(game=cf, player_id=1, max_depth=1)
    agent_one = RandomAgent(game=cf)
    # agent_two = MinMaxAgent(game=cf, player_id=2, max_depth=1)
    agent_two = RandomAgent(game=cf)
    arena = Arena(agent_one=agent_one, agent_two=agent_two, game=cf)
    winner = arena.play()
    return winner, arena.game


def main() -> None:
    n_games = int(input("How many games do you want to play?: "))
    scores = [0, 0]
    for _ in tqdm(range(n_games)):
        cf = ConnectFour()
        winner, game = generate_random_game(cf)
        scores[winner - 1] += 1
        
        # print("\n====================\n")
        # game.print_board()
      
    print(f"Score after {n_games} games.")
    print(scores)


if __name__ == "__main__":
    main()
