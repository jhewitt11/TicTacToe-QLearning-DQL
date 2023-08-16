from collections import defaultdict
import random



from game import TicTacToe
from game import greedy_pick_moves
from game import ExhaustiveSearch


'''

0   1   2

3   4   5

6   7   8

---------


x   o   x

x   o   o

-   x   -

'''

game = TicTacToe()

game.move('x', 2)
game.move('o', 7)

'''
game.move('x', 2)
game.move('o', 1)
game.move('x', 7)
game.move('o', 5)
game.move('x', 3)
'''

result = ExhaustiveSearch(game)

print(result)