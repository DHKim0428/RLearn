EMPTY = 0
X = 1
O = 4
DRAW = 3
N = 3

def withinBoard(i, j): return 0<=i<N and 0<=j<N
def opponent(player): 
    if player == X:
        return O
    return X