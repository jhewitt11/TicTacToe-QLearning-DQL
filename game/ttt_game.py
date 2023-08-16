
class TicTacToe():
    '''
    TicTacToe game class.

    This game initializes with an empty board. Moves are played
    via self.move(player, position) method.

    Some entry validation features have been added to self.move()

    For now the gameboard is represented by a nine element list. 0 (int) represents an empty position. x's are represented as 1 (int) and o's as -1 (int)

    The player argument can either be a 'x' or 'o'. This is intended to keep the interaction with the game object natural feeling.

    Instance variables are modified in the self.move() and self.check_win_condition() methods.
    '''

    def __init__(self, *, first_move_x = True):
        '''
        Initialized variables are pretty straightforward. 
        '''

        # initializations
        self.board          = [0 for x in range(9)]
        self.lookup_dict    = {'x' : 1,
                               'o' : -1,
                               1 : 'x',
                               -1 : 'o',
                               }
        self.win            = False
        self.winner         = False
        self.win_dir        = False
        self.win_ind        = False
        self.age            = 0

        # first move selection
        if first_move_x :
            self.turn = 1
        else:
            self.turn = -1

        return

    def check_win_conditions(self,):

        # 0, 1, 2
        # 3, 4, 5
        # 6, 7, 8
        # come up with list of triples to check if sum to 3 or -3
        potentials = []

        # verticals
        for i in range(3):
            potentials.append(((self.board[i] + self.board[i + 3] + self.board[i + 6]), 'v', i))

        # horizontals
        for i in range(3):
            i = i * 3
            potentials.append(((self.board[i] + self.board[i + 1] + self.board[i + 2]), 'h', i))

        # 2 diagonals
        potentials.append(((self.board[0] + self.board[4] + self.board[8]), 'd', 0))
        potentials.append(((self.board[6] + self.board[4] + self.board[2]), 'd', 1))

        for line_sum, direction, ind in potentials :
            if line_sum == 3 :
                self.win = True
                self.win_dir = direction
                self.win_ind = ind
                self.winner = 1

            elif line_sum == -3 :
                self.win = True
                self.win_dir = direction
                self.win_ind = ind
                self.winner = -1

        return self.win


    def move(self, player, position):

        num_val = self.lookup_dict.get(player)

        if self.win != False:
            print(f'Game is over')
            return 200

        if num_val != self.turn :
            print(f'Not player {str(player)}\'s turn')
            return 200 

        # check position is a valid int value
        if not (isinstance(position, int) and 0 <= position <= 9) :
            print(f'{str(position)} is not valid')
            return 200

        # check board position is available
        if self.board[position] != 0 :
            print(f'Move : {str(position)} is taken')
            return 200

        # make the change on the board
        self.board[position] = self.turn
        self.age += 1

        ## immediately check win condition
        if self.check_win_conditions():
            #print(f'Winner : {self.lookup_dict.get(self.winner)}')
            return 100

        if self.age == 9:
            #print(f'Draw!')
            self.winner = -2
            return 100

        # make the change to game turn state
        self.turn *= -1

        return 100
