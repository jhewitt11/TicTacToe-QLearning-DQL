import copy
import random
import numpy as np

from collections import defaultdict, deque, namedtuple
from game import TicTacToe


def pick_from_random(board):

    potential_positions = [x for x in range(9) if board[x] == 0]
    position = random.choice(potential_positions)

    return position


def pick_from_greedy_heuristic(player, board):

    '''
    _0_|_1_|_2_
    _3_|_4_|_5_
     6 | 7 | 8 

    Example game : 

    _x_|_o_|___
    ___|_o_|_o_
       | x | x 

    represented as ... (0's omitted)

    _1_|_-1_|___
    ___|_-1_|_-1_
       | 1  | 1


    x's turn. 
    Moves available : 2, 3, 6

    For each available position, there is an offensive and defensive score associated with the row, column and diagonal(s) it is apart of.

    position 2 :
        row : 
            off : 1
            def : -1
        col :
            off : 1
            def : -1

        diag1 :
            off : 0
            def : 0

        diag2 :
            off : 0
            def : -1

    '''

    # invert and continue if player = -1
    if player == 'o' :
        board = [x * -1 for x in board]

    potential_moves = [x for x in range(9) if board[x] == 0]

    offensive_weights = []
    defensive_weights = []

    for move in potential_moves :

        col_ind = move % 3
        row_ind = int(move / 3)

        # row component (offensive and defensive)
        row_indices = [x + (row_ind  * 3) for x in range(3)]
        
        row_offensive_component = sum([board[x] for x in row_indices if board[x] == 1])
        row_defensive_component = sum([board[x] for x in row_indices if board[x] == -1])

        # col component
        col_indices = [col_ind + (x * 3) for x in range(3)]

        col_offensive_component = sum([board[x] for x in col_indices if board[x] == 1])
        col_defensive_component = sum([board[x] for x in col_indices if board[x] == -1])


        # diagonal 1
        diag1_indices = [0,  4, 8]

        if move in diag1_indices :
            diag1_offensive_component = sum([board[x] for x in diag1_indices if board[x] == 1])
            diag1_defensive_component = sum([board[x] for x in diag1_indices if board[x] == -1])
        else :
            diag1_offensive_component = 0
            diag1_defensive_component = 0

       # diagonal 2
        diag2_indices = [2, 4, 6]

        if move in diag2_indices :
            diag2_offensive_component = sum([board[x] for x in diag2_indices if board[x] == 1])
            diag2_defensive_component = sum([board[x] for x in diag2_indices if board[x] == -1])
        else :
            diag2_offensive_component = 0
            diag2_defensive_component = 0

        offensive_score = max([row_offensive_component,
                            col_offensive_component,
                            diag1_offensive_component,
                            diag2_offensive_component])

        defensive_score = min([row_defensive_component,
                            col_defensive_component,
                            diag1_defensive_component,
                            diag2_defensive_component])


        offensive_weights.append(offensive_score)
        defensive_weights.append(defensive_score * -1)

    # 
    max_offensive_score = max(offensive_weights)
    offensive_best_index = offensive_weights.index(max_offensive_score)

    # 
    max_defensive_score = max(defensive_weights)
    defensive_best_index = defensive_weights.index(max_defensive_score)

    if max_offensive_score >= max_defensive_score :
        final_index = offensive_best_index
    else:
        final_index = defensive_best_index

    return potential_moves[final_index]


def pick_from_Q_table(player, q_table, board):

    if player == 'o' :
        board = [x * -1 for x in board]

    board2 = board[:]
    action_vals = q_table[tuple(board)]

    ''' selection code from training fx'''
    
    # mask taken moves with -200
    moves_masked = [x if board[i] == 0 else -200 for i, x in enumerate(action_vals)]

    #Its not enough to find the max value.
    #Ties need to be broken randomly to promote exploration.
    if max(moves_masked) == 0 :
        zero_inds = [i for i, x in enumerate(moves_masked) if moves_masked[i] == 0]
        max_value_action_ind = random.choice(zero_inds)
        bundle = None

    else :
        max_value_action_ind = int(np.argmax(moves_masked))  
        board2[max_value_action_ind] = 1

        bundle = (
            tuple(board),                                  #state
            max_value_action_ind,                   #action
            tuple(board2),                                 #next state
            moves_masked[max_value_action_ind]      #reward
            )

        assert(bundle[0] != bundle[2])

    return max_value_action_ind, bundle


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def ExhaustiveSearch(ttt):
    '''
    ttt is type TicTacToe()

    This function takes in a game with a board state and evaluates the game space moving forward. All possible outcomes are recorded and resulting Win / Loss / Draw tallies are returned.

    ** ExhaustiveSearch ** right now is a tool - not yet a policy

    '''
    '''
    Exhaustive search

    game state - x
    possible moves - 1, 2, 3

        child node 1
            play out game scenarios.
            tally results - these define node's strength / utility
            use to compare.

        child node 2
            ...
        child node 3
            ...

    Board state
        Explore
            - calculate moves
            
            - explore moves
                - end state? win / lose / draw
                - report

            - take totality of results and use this to..
                - chose the best route
                - 
    '''
    # base case
    # check final state - win? lose? draw?

    if ttt.winner == 1:
        return {
            tuple(ttt.board) : {
                '1' : 1,
                '-1' : 0,
                'draw' : 0
            }
        }
    elif ttt.winner == -1:
        return {
            tuple(ttt.board) : {
                '1' : 0,
                '-1' : 1,
                'draw' : 0
            }
        }
    elif ttt.winner == -2:
        return {
            tuple(ttt.board) : {
                '1' : 0,
                '-1' : 0,
                'draw' : 1
            }
        }

    # this is where a move policy would be added.
    possible_moves = [x for x in range(9) if ttt.board[x] == 0]

    child_results = []
    for position in possible_moves :
        ttt_child = copy.deepcopy(ttt)
        result = ttt_child.move( ttt_child.lookup_dict[ttt_child.turn] , position)
        child_results.append(ExhaustiveSearch(ttt_child))

    p1 = 0
    pn1 = 0
    draw = 0

    for child_result in child_results :

        scores = list(child_result.values())[0]

        p1 += scores['1']
        pn1 += scores['-1']
        draw += scores['draw']


    return {
        tuple(ttt.board):{
            '1' : p1,
            '-1' : pn1,
            'draw' : draw
        }

    }


def state_factory():
        return [0,0,0,0,0,0,0,0,0]


def train_Q_table(rounds, learning_rate, discount_factor, greedy_ratio, competitor_policy = 'random'):

    q_table = defaultdict(state_factory)

    for i in range(rounds) :

        first_move_x = (i%2 == 0)
        ttt = TicTacToe(first_move_x = first_move_x)

        while ttt.win == False and ttt.age < 9 :

            if ttt.turn == -1 :
                if competitor_policy == 'random' :
                    position = pick_from_random(ttt.board)

                elif competitor_policy == 'greedy' :
                    position = pick_from_greedy_heuristic(player = -1, board = ttt.board)

                result = ttt.move('o', position)

            else :
                player = 'x'
                s0 = tuple(ttt.board)
                action_vals = q_table[s0]

                valid_ind_vals = [(i, x) for i, x in enumerate(action_vals) if ttt.board[i] == 0]

                if random.random() > greedy_ratio :
                    max_value_action_ind, _ = random.choice(valid_ind_vals)

                else:
                    valid_ind_vals.sort(reverse = True, key = lambda x : x[1])

                    if valid_ind_vals[0][1] == 0:
                        zero_inds = [i for i, x in valid_ind_vals if x == 0]
                        max_value_action_ind = random.choice(zero_inds)

                    else:
                        max_value_action_ind = valid_ind_vals[0][0]


                # The move is entered into the game engine 
                result = ttt.move(player, max_value_action_ind)
                
                # This is the new current state of the board.
                s1 = tuple(ttt.board)


                if ttt.win == True and ttt.winner == 1 :
                    reward = 1

                elif ttt.win == True and ttt.winner == -1 :
                    reward = -1

                elif ttt.age == 9 :
                    reward = 0.75
                else : 
                    reward = 0

                q_vals_next_state = q_table[s1]
                q_curr = q_table[s0][max_value_action_ind] 


                piece1 = (1 - learning_rate) * q_curr
                piece2 = learning_rate * (reward + discount_factor * max(q_vals_next_state))
                q_curr = piece1 + piece2

                q_table[s0][max_value_action_ind] = q_curr

            assert(result == 100)

    return q_table

