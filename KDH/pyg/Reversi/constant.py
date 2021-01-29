EMPTY = 0
N = 8
B = 1
W = 2
DRAW = 3
VALID = 4

def withinBoard(i, j): return 0<=i<N and 0<=j<N
def opponent(player): 
    if player == B:
        return W
    return B