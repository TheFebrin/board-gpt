import streamlit as st
import pickle
import os
import torch
from connect_four.connect_four import ConnectFour
from connect_four.agents import Agent, MinMaxAgent, GPTAgent
from mingpt.model import GPT, GPTConfig
from connect_four.connect_four_dataset import ConnectFourDataset, CharConnectFourDataset


def create_board():
    board = ConnectFour()
    return board


def create_agent(board: ConnectFour, agent: str):
    if agent == "MinMax":
        agent = MinMaxAgent(board, max_depth=3, player_id=2)
    else:
        print("Files: ", os.listdir())
        print("Files board-gpt: ", os.listdir("board-gpt"))
        with open("board-gpt/minmax_biggest_dataset_100970.pkl", "rb") as f:
            minmax_games = pickle.load(f)
        minmax_cf_data = ConnectFourDataset(data_size=0, train_size=7138, games_to_use=minmax_games)
        minmax_char_cf_dataset = CharConnectFourDataset(minmax_cf_data)
        mconf = GPTConfig(
            minmax_char_cf_dataset.config.vocab_size, minmax_char_cf_dataset.config.block_size, n_layer=8, n_head=8,
            n_embd=512
        )
        model = GPT(mconf).to("cpu")
        model.load_state_dict(torch.load("board-gpt/BIG_V3_gpt_at_20230620_213816.ckpt"))
        agent = GPTAgent(
            model=model,
            game=board,
            preprocessing_config=minmax_char_cf_dataset.config,
            first_move=4,
            name="MinMax GPT",
            device="cpu",
        )
    return agent


def display_board(board):
    st.table(board.board)


def handle_moves(board, agent):
    cols = st.columns(7)
    with cols[0]:
        col0 = st.button("0", key="0")

    with cols[1]:
        col1 = st.button("1", key="1")

    with cols[2]:
        col2 = st.button("2", key="2")

    with cols[3]:
        col3 = st.button("3", key="3")

    with cols[4]:
        col4 = st.button("4", key="4")

    with cols[5]:
        col5 = st.button("5", key="5")

    with cols[6]:
        col6 = st.button("6", key="6")

    column = [col0, col1, col2, col3, col4, col5, col6, True].index(True)

    if column < 7:
        st.text(f"Your last move: {column}")
        if board.is_move_possible(column):
            board.make_move(1, column)
            check_game_status(board)
            if board.is_game_finished() == -1:
                ai_move = agent.choose_move()
                board.make_move(2, ai_move)
                check_game_status(board)
                st.write(f"AI move: {ai_move}")
        else:
            st.text("Invalid move, please try again.")


def check_game_status(board):
    if board.is_game_finished() != -1:
        if board.is_game_finished() == 1:
            st.write("Congratulations, you won!")
        elif board.is_game_finished() == 2:
            st.write("AI won.")
        else:
            st.write("It's a draw.")
        if st.button("Play Again"):
            st.session_state["board"] = create_board()
            st.session_state["agent"] = create_agent(st.session_state["board"])


def main():
    st.title("Play Connect Four vs AI")
    agent = st.selectbox(f"Select AI and restart game after: ", ["MinMax", "GPT"])
    if "agent" in st.session_state:
        st.text(f"Current AI: {st.session_state['agent']}")

    if st.button("Restart game"):
        st.session_state["board"] = create_board()
        st.session_state["agent"] = create_agent(
            board=st.session_state["board"], agent=agent
        )

    if "board" not in st.session_state:
        st.session_state["board"] = create_board()
    if "agent" not in st.session_state:
        st.session_state["agent"] = create_agent(
            board=st.session_state["board"], agent=agent
        )

    handle_moves(st.session_state.board, st.session_state.agent)
    display_board(st.session_state.board)


if __name__ == "__main__":
    main()
