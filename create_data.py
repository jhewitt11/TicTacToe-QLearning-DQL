from collections import defaultdict
import random
import sys
import pickle as pkl
import random

from game import TicTacToe
from game import train_Q_table
from game import state_factory

from game import pick_from_random
from game import pick_from_greedy_heuristic
from game import pick_from_Q_table

trial_rounds        = 500000
competitor_policy   = 'random'
training_rounds     = 500000

q_learning_rate     = 0.35
q_discount_factor   = 0.5
q_greedy_ratio      = 0.35

TRAIN_TABLE = True
WRITE_MEMORY = True

ID = random.randint(1000,9999)

if TRAIN_TABLE :
    q_table = train_Q_table(rounds = training_rounds, 
                            learning_rate = q_learning_rate,
                            discount_factor = q_discount_factor,
                            greedy_ratio = q_greedy_ratio,
                            competitor_policy = competitor_policy,
                            )

    print(f'Trial Information :')
    print(f'\tTraining rounds : {training_rounds}')
    print(f'\tCompetitor training policy : {competitor_policy}')
    print(f'\n\tQ Learning Rate : {q_learning_rate}')
    print(f'\tQ Discount Factor : {q_discount_factor}')
    print(f'\tQ Greedy Ratio : {q_greedy_ratio}\n')


    table_dict = {
        'training_rounds'   : training_rounds,    
        'q_learning_rate'   : q_learning_rate,
        'q_discount_factor' : q_discount_factor,
        'q_greedy_ratio'    : q_greedy_ratio,
        'competitor_policy' : competitor_policy,
        'q_table'           : q_table,
        'ID'                : ID
    }

    with open('data/q_table_'+str(ID)+'.pkl', 'wb') as file:
        pkl.dump(table_dict, file)    

else :
    with open('data/q_table_.pkl', 'rb') as file:
        table_dict = pkl.load(file)

    q_table = table_dict['q_table']



'''
Trial Portion

'''
memory = []
x_turns = 0
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
for i in range(trial_rounds) :

    # make a new game and alternate who goes first
    first_move_x = (i%2 == 0)
    ttt = TicTacToe(first_move_x = first_move_x)

    # play the game out
    while ttt.win == False and ttt.age < 9 :

        if ttt.turn == 1 :
            player = 'x'
            x_turns += 1
            # Policy Selection
            position, bundle = pick_from_Q_table(player, q_table, ttt.board)
            
            if bundle != None:
                memory.append(bundle)
                

        else :
            player = 'o'

            # Policy Selection
            position = pick_from_random(ttt.board)
            #position = pick_from_greedy_heuristic(player, ttt.board)
            #position = pick_from_Q_table(player, q_table, ttt.board)
            #position = pick_from_DQN(player, ttt.board, validate_move = True)

        result = ttt.move(player, position)
        assert(result == 100)

    # keep track of stats
    if ttt.winner == -2 :
        stats[ttt.winner] += 1
    else:
        stats[ttt.winner][ttt.win_dir] += 1
        stats[ttt.winner]['win_lengths'][ttt.age] += 1
                                                                                    

stats[1]['total'] = sum([x for x in list(stats[1].values()) if isinstance(x, int)])
stats[-1]['total'] = sum([x for x in list(stats[-1].values()) if isinstance(x, int)])

print(f'\n\tTrial Rounds : {trial_rounds}')
print(f'\tPlayer 1 wins : {stats[1]["total"]}\t ({stats[1]["total"] / trial_rounds : .3})\t {stats[1]["win_lengths"]}')
print(f'\tPlayer -1 wins : {stats[-1]["total"]}\t ({stats[-1]["total"] / trial_rounds : .3})\t {stats[-1]["win_lengths"]}')
print(f'\tDraws : {stats[-2]}\t\t ({stats[-2] / trial_rounds : .3})')
print(f'\t# of player X turns : {x_turns}\n\t# of player x moves recorded : {len(memory)}\n\n')


for i in range(len(memory)):
    assert(memory[i][0] != memory[i][2])


if WRITE_MEMORY :
    with open('data/memory_'+str(ID)+'.pkl', 'wb') as file:
        pkl.dump(memory, file)