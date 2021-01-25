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
        self.cell_row_sum = [EMPTY]*N
        self.cell_column_sum = [EMPTY]*N
        self.cell_skew_sum = [EMPTY]*2
        self.num_turn = 0
        self.canvas = canvas

    # def toState(self):
    #     S

    def winner(self):
        for s in self.cell_row_sum:
            if s == X * N: return X
            if s == O * N: return O
        for s in self.cell_column_sum:
            if s == X * N: return X
            if s == O * N: return O
        for s in self.cell_skew_sum:
            if s == X * N: return X
            if s == O * N: return O
        return DRAW

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
        return self.isempty(i, j)

    def setMove(self, player, move):
        i, j = move
        self.cell[i][j] = player
        self.cell_row_sum[i] += player
        self.cell_column_sum[j] += player
        self.cell_skew_sum[0] += player if i == j else 0
        self.cell_skew_sum[1] += player if i + j + 1 == N else 0

        self.num_turn += 1
        if self.canvas != None:
            self.canvas.addMark(i, j, player, self.num_turn)
            if self.winner() == player:
                self.canvas.setGameOver(player)

    def copy(self):
        b = Board()
        b.cell = copy.deepcopy(self.cell)
        b.cell_row_sum = copy.deepcopy(self.cell_row_sum)
        b.cell_column_sum = copy.deepcopy(self.cell_column_sum)
        b.cell_skew_sum = copy.deepcopy(self.cell_skew_sum)
        return b

    def boardAfterMove(self, player, move):
        b = self.copy()
        b.setMove(player, move)
        return b

    def value(self, player):
        result = self.winner()
        if result == DRAW: return 0
        if result == player: return 10
        return -10

    def noValidMoves(self, player):
        return len(self.validMoves(player)) == 0