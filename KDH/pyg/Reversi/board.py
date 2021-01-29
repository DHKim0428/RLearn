# from constant import *
from graphics import *
# import pygame
import random
import copy

class Board:
    def isempty(self, i, j): return self.cell[i][j] == EMPTY
    def mark(self, i, j): return self.cell[i][j]

    def __init__(self, state=None, canvas=None):
        self.cell = [[EMPTY]*N for i in range(N)]
        self.num_turn = 0
        self.canvas = canvas
        self.cell[N//2 - 1][N//2 - 1] = W
        self.cell[N//2 - 1][N//2] = B
        self.cell[N//2][N//2 - 1] = B
        self.cell[N//2][N//2] = W
        # W B
        # B W

    # def toState(self):
    #     S

    def winner(self):
        black_num = 0
        white_num = 0
        for i in range(N):
            for j in range(N):
                if self.cell[i][j] == B:
                    black_num += 1
                elif self.cell[i][j] == W:
                    white_num += 1

        # assert black_num + white_num == N * N
        if black_num == white_num: return DRAW
        if black_num > white_num: return B
        return W

    def validMoves(self, player):
        L = []
        for i in range(N):
            for j in range(N):
                move = (i, j)
                if self.isValidMove(player, move):
                    L.append(move)
        random.shuffle(L)
        return L

    def isValidMove(self, player, move):
        i, j = move
        o_player = opponent(player)
        D = [ (-1, -1), (-1, 0), (-1, 1),
                (0, -1),         (0, 1),
                (1, -1), (1, 0), (1, 1)]
        if not self.isempty(i, j):
            if self.cell[i][j] != VALID:
                return False

        for (di, dj) in D:
            p, q = i+di, j+dj
            if (not withinBoard(p, q)) or self.cell[p][q] != o_player:
                continue
            while withinBoard(p, q) and self.cell[p][q] == o_player:
                p += di
                q += dj
            if (not withinBoard(p, q)) or self.cell[p][q] != player:
                continue
            return True
        return False
            
    def setMove(self, player, move):
        i, j = move
        o_player = opponent(player)
        D = [ (-1, -1), (-1, 0), (-1, 1),
                (0, -1),         (0, 1),
                (1, -1), (1, 0), (1, 1)]
        self.cell[i][j] = player

        for (di, dj) in D:
            p, q = i+di, j+dj
            if (not withinBoard(p, q)) or self.cell[p][q] != o_player:
                continue
            while withinBoard(p, q) and self.cell[p][q] == o_player:
                p += di
                q += dj
            if (not withinBoard(p, q)) or self.cell[p][q] != player:
                continue
            # 이 방향으로는 뒤집을 수 있음
            p, q = i+di, j+dj
            while self.cell[p][q] == o_player:
                self.cell[p][q] = player
                p += di
                q += dj
        
        self.num_turn += 1
        if self.canvas != None:
            self.canvas.addMark(copy.deepcopy(self.cell))
            # if self.winner() == player:
            #     self.canvas.setGameOver(player)

    def copy(self):
        b = Board()
        b.cell = copy.deepcopy(self.cell)
        return b

    def boardAfterMove(self, player, move):
        b = self.copy()
        b.setMove(player, move)
        return b

    # greedy method
    def value(self, player):
        player_num = 0 
        o_player_num = 0
        for i in range(N):
            for j in range(N):
                if self.cell[i][j] == player:
                    player_num += 1
                elif self.cell[i][j] == opponent(player):
                    o_player_num += 1

        if player_num == 0:
            return -100

        # if player_num == o_player_num:
        #     return 0
        # if player_num + o_player_num == N*N:
        #     return 100 if player_num > o_player_num else -100
        
        return player_num - o_player_num

    def noValidMoves(self, player):
        return len(self.validMoves(player)) == 0

    def drawValidMoves(self, player):
        for i in range(N):
            for j in range(N):
                if self.isValidMove(player, (i, j)):
                    self.cell[i][j] = VALID
        self.canvas.draw()
    