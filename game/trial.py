from game import  TicTacToe

from game import pick_from_random
from game import pick_from_greedy_heuristic
from game import pick_from_Q_table

import pickle as pkl
import pandas as pd


def policy_selection(policy, model_id, player):

    if policy == 'random':
        return pick_from_random

    elif policy == 'greedy':

        def inner(board):
            return pick_from_greedy_heuristic(player, board)
        
        return inner

    elif policy == 'q_table' :

        with open('models_data/q_table_' + str(model_id) + '.pkl', 'rb') as file:
            q_dict = pkl.load(file)

        q_table = q_dict['q_table']

        def inner(board):
            return pick_from_Q_table(player, q_table, board)[0]

        return inner 

    else : 
        raise ValueError('Unknown Policy Type')

class ttt_trial():

    def __init__(self, parameters):
        self.rounds   = parameters['rounds']
        self.policy_x = policy_selection(parameters['policy_x'], parameters['model_id_x'], 1)
        self.policy_o = policy_selection(parameters['policy_o'], parameters['model_id_o'], -1)
        self.stats = None
        self.parameters = parameters

    def run(self, ):
        
        stats = { 
                    1 : {
                            'h' : 0,
                            'v' : 0,
                            'd' : 0,
                            'win_lengths' : [0 for x in range(10)]
                        },
                    -1 : {
                            'h' : 0,
                            'v' : 0,
                            'd' : 0,
                            'win_lengths' : [0 for x in range(10)]
                        },
                    -2 : 0}

        for i in range(self.rounds):
            # make a new game and alternate who goes first
            first_move_x = (i%2 == 0)
            ttt = TicTacToe(first_move_x = first_move_x)

            # play the game out
            while ttt.win == False and ttt.age < 9 :

                if ttt.turn == 1 :
                    player = 'x'
                    position = self.policy_x(ttt.board)

                else :
                    player = 'o'
                    position = self.policy_o(ttt.board)

                result = ttt.move(player, position)

            # keep track of stats
            if ttt.winner == -2 :
                stats[ttt.winner] += 1
            else:
                stats[ttt.winner][ttt.win_dir] += 1
                stats[ttt.winner]['win_lengths'][ttt.age] += 1

        self.stats = stats

        return 200

    def get_result_dict(self, ):

        if self.stats == None : raise ValueError('Trial stats not created.')

        '''
        Make a pandas dataframe.
        '''

        player1_total = self.stats[1]['h'] + self.stats[1]['v'] + self.stats[1]['d']
        player0_total = self.stats[-1]['h'] + self.stats[-1]['v'] + self.stats[-1]['d']

        data = {
            'rounds'                    :   self.rounds,
            'player1 wins'              :   player1_total,
            'player0 wins'              :   player0_total,
            'draws'                     :   self.stats[-2],

            'player1 policy'            :   self.parameters['policy_x'],
            'player1 model id'          :   self.parameters['model_id_x'],

            'player0 policy'            :   self.parameters['policy_o'],
            'player0 model id'          :   self.parameters['model_id_o'],

            'player1 horiztonal wins'   :    self.stats[1]['h'],
            'player1 vertical wins'     :    self.stats[1]['v'],
            'player1 diagonal wins'     :    self.stats[1]['d'],

            'player0 horiztonal wins'   :   self.stats[-1]['h'],
            'player0 vertical wins'     :   self.stats[-1]['v'],
            'player0 diagonal wins'     :   self.stats[-1]['d'],
        }

        return data