import chess
import numpy as np

#Main file that does the sim and organism work

base_pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

base_knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]

base_bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]

base_rookstable = [
    0, 0, 0, 10, 10, 0, 0, 0,
    -5, 5, 5, 5, 5, 5, 5, -5,
    -5, 5, 5, 5, 5, 5, 5, -5,
    -5, 5, 5, 5, 5, 5, 5, -5,
    -5, 5, 5, 5, 5, 5, 5, -5,
    -5, 5, 5, 5, 5, 5, 5, -5,
    10, 20, 20, 20, 20, 20, 20, 10,
    0, 0, 0, 0, 0, 0, 0, 0]

base_queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

base_kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]

table_len = 64

def turnary_table_encoding(board):
    # table is a 1D np array of length 64
    # chess_board is a chess.Board() object
    # returns a 1D np array of length 64
    # Each element of the array is either -1, 0, or 1
    # -1 represents an enemy piece
    # 0 represents an empty square
    # 1 represents a friendly piece
    #output is a 1D np array of length # of squares on the board (64) times the number of pieces (6)

    if board.turn:
        turn = 1
    else:
        turn = -1


    temp_pawn = np.zeros(64)

    for i in board.pieces(chess.PAWN, chess.WHITE):
        temp_pawn[i] = 1 * turn
    for i in board.pieces(chess.PAWN, chess.BLACK):
        temp_pawn[i] = -1 * turn

    temp_knight = np.zeros(64)
    
    for i in board.pieces(chess.KNIGHT, chess.WHITE):
        temp_knight[i] = 1 * turn
    for i in board.pieces(chess.KNIGHT, chess.BLACK):
        temp_knight[i] = -1 * turn

    temp_bishop = np.zeros(64)

    for i in board.pieces(chess.BISHOP, chess.WHITE):
        temp_bishop[i] = 1 * turn
    for i in board.pieces(chess.BISHOP, chess.BLACK):
        temp_bishop[i] = -1 * turn

    temp_rook = np.zeros(64)

    for i in board.pieces(chess.ROOK, chess.WHITE):
        temp_rook[i] = 1 * turn
    for i in board.pieces(chess.ROOK, chess.BLACK):
        temp_rook[i] = -1 * turn

    temp_queen = np.zeros(64)

    for i in board.pieces(chess.QUEEN, chess.WHITE):    
        temp_queen[i] = 1 * turn
    for i in board.pieces(chess.QUEEN, chess.BLACK):
        temp_queen[i] = -1 * turn

    temp_king = np.zeros(64)

    for i in board.pieces(chess.KING, chess.WHITE):
        temp_king[i] = 1 * turn
    for i in board.pieces(chess.KING, chess.BLACK):
        temp_king[i] = -1 * turn

    return np.concatenate((temp_pawn, temp_knight, temp_bishop, temp_rook, temp_queen, temp_king), axis=0)

    

    

input_pawn_table = np.zeros((1, 64))
input_knights_table = np.zeros((1, 64))
input_bishop_table = np.zeros((1, 64))
input_rooks_table = np.zeros((1, 64))
input_queen_table = np.zeros((1, 64))
input_king_table = np.zeros((1, 64))
#This allows the organism to learn a table of values for each piece
#The tables are used to evaluate the board state and make decisions
#The values are learned through the organism's neural network
# The tables must be the first layer of the neural network
# TODO make sure that these tables are the first layer of the neural network

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

    pawnsq = sum([base_pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-base_pawntable[chess.square_mirror(i)] for i in board.pieces(chess.PAWN, chess.BLACK)])

    knightsq = sum([base_knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-base_knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])

    bishopsq = sum([base_bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-base_bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])

    rooksq = sum([base_rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-base_rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])

    queensq = sum([base_queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-base_queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])

    kingsq = sum([base_kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-base_kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])

    if organism == None:
        eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
    else:
        embedding = turnary_table_encoding(board)
        eval = organism.predict(embedding.reshape((1, -1)))#.reshape((1, -1))

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
    # alphabeta pruning to find the best move using minimax
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
                move = selectmove(1, board, organism_2)
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