import chess
import numpy as np
import dill
import copy

#Main file that does the sim and organism work

pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]

bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]

rookstable = [
    0, 0, 0, 10, 10, 0, 0, 0,
    -5, 5, 5, 5, 5, 5, 5, -5,
    -5, 5, 5, 5, 5, 5, 5, -5,
    -5, 5, 5, 5, 5, 5, 5, -5,
    -5, 5, 5, 5, 5, 5, 5, -5,
    -5, 5, 5, 5, 5, 5, 5, -5,
    10, 20, 20, 20, 20, 20, 20, 10,
    0, 0, 0, 0, 0, 0, 0, 0]

queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]


def evaluate_board(board, organism=None):
    if board.is_checkmate():
        if board.turn:
            return -9999
        else:
            return 9999
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])

    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])

    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])

    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])

    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])

    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])

    if organism == None:
        eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
    else:
        eval = organism.predict(
            np.array([material, pawnsq, knightsq, bishopsq, rooksq, queensq, kingsq]).reshape((1, -1)))

        # compare performance of below
        #         if type(eval) is np.ndarray:
        #             eval = eval[0][0]

        eval = np.array(eval).flatten()
        eval = eval[0]

    if board.turn:
        return eval
    else:
        return -eval

def selectmove(depth, board, organism=None):
    bestMove = chess.Move.null()
    bestValue = -99999
    alpha = -100000
    beta = 100000
    for move in board.legal_moves:
        board.push(move)
        boardValue = -alphabeta(-beta, -alpha, depth - 1, board, organism)
        if boardValue > bestValue:
            bestValue = boardValue
            bestMove = move
        if (boardValue > alpha):
            alpha = boardValue
        board.pop()
    return (bestMove, bestValue)


def alphabeta(alpha, beta, depthleft, board, organism=None):
    bestscore = -9999
    if (depthleft == 0):
        return quiesce(alpha, beta, board, organism)
    for move in board.legal_moves:
        board.push(move)
        score = -alphabeta(-beta, -alpha, depthleft - 1, board, organism)
        board.pop()
        if (score >= beta):
            return score
        if (score > bestscore):
            bestscore = score
        if (score > alpha):
            alpha = score
    return bestscore

def quiesce(alpha, beta, board, organism=None):
    stand_pat = evaluate_board(board, organism)
    if (stand_pat >= beta):
        return beta
    if (alpha < stand_pat):
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiesce(-beta, -alpha, board, organism)
            board.pop()

            if (score >= beta):
                return beta
            if (score > alpha):
                alpha = score
    return alpha


class Organism:
    def __init__(self, dimensions, use_bias=True, output='softmax'):
        self.score = 0

        self.winner = False

        self.layers = []
        self.biases = []
        self.use_bias = use_bias
        self.output = self._activation(output)
        self.dimensions = dimensions
        for i in range(len(dimensions) - 1):
            shape = (dimensions[i], dimensions[i + 1])
            std = np.sqrt(2 / sum(shape))
            layer = np.random.normal(0, std, shape)
            bias = np.random.normal(0, std, (1, dimensions[i + 1])) * use_bias
            self.layers.append(layer)
            self.biases.append(bias)

    def _activation(self, output):
        if output == 'softmax':
            return lambda X: np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
        if output == 'sigmoid':
            return lambda X: (1 / (1 + np.exp(-X)))
        if output == 'linear':
            return lambda X: X
        if output == 'relu':
            return lambda X: max(0, X)

    def predict(self, X):
        if not X.ndim == 2:
            raise ValueError(f'Input has {X.ndim} dimensions, expected 2')
        if not X.shape[1] == self.layers[0].shape[0]:
            raise ValueError(f'Input has {X.shape[1]} features, expected {self.layers[0].shape[0]}')
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            X = X @ layer + np.ones((X.shape[0], 1)) @ bias
            if index == len(self.layers) - 1:
                X = self.output(X)  # output activation
            else:
                X = np.clip(X, 0, np.inf)  # ReLU

        return X

    def predict_choice(self, X, deterministic=True):
        probabilities = self.predict(X)
        if deterministic:
            return np.argmax(probabilities, axis=1).reshape((-1, 1))
        if any(np.sum(probabilities, axis=1) != 1):
            raise ValueError(f'Output values must sum to 1 to use deterministic=False')
        if any(probabilities < 0):
            raise ValueError(f'Output values cannot be negative to use deterministic=False')
        choices = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            U = np.random.rand(X.shape[0])
            c = 0
            while U > probabilities[i, c]:
                U -= probabilities[i, c]
                c += 1
            else:
                choices[i] = c
        return choices.reshape((-1, 1))

    def mutate(self, stdev=0.03):
        for i in range(len(self.layers)):
            self.layers[i] += np.random.normal(0, stdev, self.layers[i].shape)
            if self.use_bias:
                self.biases[i] += np.random.normal(0, stdev, self.biases[i].shape)

    def mate(self, other, mutate=True):
        if self.use_bias != other.use_bias:
            raise ValueError('Both parents must use bias or not use bias')
        if not len(self.layers) == len(other.layers):
            raise ValueError('Both parents must have same number of layers')
        if not all(self.layers[x].shape == other.layers[x].shape for x in range(len(self.layers))):
            raise ValueError('Both parents must have same shape')

        child = copy.deepcopy(self)
        for i in range(len(child.layers)):
            pass_on = np.random.rand(1, child.layers[i].shape[1]) < 0.5
            child.layers[i] = pass_on * self.layers[i] + ~pass_on * other.layers[i]
            child.biases[i] = pass_on * self.biases[i] + ~pass_on * other.biases[i]
        if mutate:
            child.mutate()
        return child

    def save_human_readable(self, filepath):
        file = open(filepath, 'w')
        file.write('----------NEW MODEL----------\n')
        file.write('DIMENSIONS\n')
        for dimension in self.dimensions:
            file.write(str(dimension) + ',')
        file.write('\nWEIGHTS\n')
        for layer in self.layers:
            file.write('NEW LAYER\n')
            for node in layer:
                for weight in node:
                    file.write(str(weight) + ',')
                file.write('\n')
            file.write('\n')
        if self.use_bias:
            file.write('BIASES:\n')
            for layer in self.biases:
                file.write('\nNEW LAYER\n')
                for connection in layer:
                    file.write(str(connection) + ',')
        file.close()

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as file:
            organism = dill.load(file)
        return organism

def captured_piece(board, move, scale=10):
    piece = None

    if board.is_capture(move):
        if board.is_en_passant(move):
            piece = chess.PAWN
        else:
            piece = board.piece_at(move.to_square).piece_type

    if piece is not None:
        if piece is chess.PAWN:
            return scale
        elif piece is chess.KNIGHT or piece is chess.BISHOP:
            return 3 * scale
        elif piece is chess.ROOK:
            return 5 * scale
        elif piece is chess.QUEEN:
            return 9 * scale
    else:
        return 0


def simulate_and_evaluate(organism_1, organism_2, print_game=False, trials=1):
    board = chess.Board()
    
    total_points_player_1 = 0
    total_points_player_2 = 0
    winner_points = 0
    loser_points = 0
    
    #The points you get for winning
    WON_POINTS = 10000000
    
    for i in range(trials):
        
        count = 0

        if print_game:
            print("###################################### STARTING NEW GAME ###########################################")
        while not board.is_game_over(claim_draw=False):
            if board.turn:
                count += 1
                if(print_game):
                    print(f'\n{count}]\n')
                #FIRST turn
                move = selectmove(3, board, organism_1)
                total_points_player_1 = total_points_player_1 + move[1]
                
                total_points_player_1 = total_points_player_1 + captured_piece(board, move[0], scale=200)
                
            
                if(print_game):
                    print("Player 1 move")
                    print(move[0])
                board.push(move[0])
                if print_game:
                    print(board)
                    print()
            else:
                #second turn
                move = selectmove(3, board, organism_2)
                total_points_player_2 = total_points_player_2 + (-1 * move[1])
                
                total_points_player_2 = total_points_player_2 + captured_piece(board, move[0], scale=200)
                
                if(print_game):
                    print("Player 2 move")
                    print(move[0])
                board.push(move[0])
                if print_game:
                    print(board)
            if print_game:
                print(board.outcome())
                print(total_points_player_1, " player 1 points found so far")
                print(total_points_player_2, " player 2 points found so far")
                
    if board.outcome().result() != "1/2-1/2":
        if board.outcome().result() == "1-0" :
            organism_1.winner = True
            print("Organism 1 won")
            total_points_player_1 = total_points_player_1 + WON_POINTS
        else:
            organism_2.winner = True
            print("Organism 2 won")
            total_points_player_2 = total_points_player_2 + WON_POINTS
    else:
        print("Draw")
        
    # print([total_points_player_1 / trials, total_points_player_2 / trials])
    organism_1.score = total_points_player_1
    organism_2.score = total_points_player_2
    return [total_points_player_1 / trials, total_points_player_2 / trials]
