from collections import defaultdict
import random

from game import TicTacToe
from game import greedy_pick_moves
from game import train_Q_table
from game import pick_from_Q_table


q_table = train_Q_table(rounds = 500000, competitor_policy = 'greedy')

for board, actions in q_table.items():
    print(f"board : {board} \t\t action scores : {[f'{num:.2f}' for num in actions]}")