EMPTY = 0
N = 5
X = 1
O = N+1
DRAW = 3

def withinBoard(i, j): return 0<=i<N and 0<=j<N
def opponent(player): 
    if player == X:
        return O
    return X