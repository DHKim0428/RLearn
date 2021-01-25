from constant import *
import pygame
import sys

class Canvas:
    def __init__(self, screen, width=600, height=600):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.large_font = pygame.font.SysFont('Noto Sans CJK HK', 72)
        self.CELL_SIZE = 200
        self.screen = screen
        self.SCREEN_WIDTH = width
        self.SCREEN_HEIGHT = height

        self.grid = [[' '] * N for i in range(N)]
        self.turn = 0
        self.game_over = 0
    
    def get_events(self):
        # return (-2, -2) if we need to close window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return (-2, -2)
            if event.type == pygame.MOUSEBUTTONDOWN and self.game_over == 0:
                column_index = event.pos[0] // self.CELL_SIZE
                row_index = event.pos[1] // self.CELL_SIZE
                print(row_index, column_index)
                return (row_index, column_index)
        return (-1, -1)
    
    def addMark(self, row, col, player, nTurn):
        self.grid[row][col] = 'O' if player == O else 'X'
        self.turn += 1
    
    def setGameOver(self, result):
        self.game_over = result

    def draw(self):
        self.screen.fill(self.BLACK)
        for row in range(N):
            for col in range(N):
                pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), 1)
                mark = self.grid[row][col]
                if mark == 'X':
                    X_image = self.large_font.render('X', True, self.YELLOW)
                    self.screen.blit(X_image, X_image.get_rect(centerx=col * self.CELL_SIZE + self.CELL_SIZE // 2,
                                                            centery=row * self.CELL_SIZE + self.CELL_SIZE // 2))
                elif mark == 'O':
                    O_image = self.large_font.render('O', True, self.WHITE)
                    self.screen.blit(O_image, O_image.get_rect(centerx=col * self.CELL_SIZE + self.CELL_SIZE // 2,
                                                            centery=row * self.CELL_SIZE + self.CELL_SIZE // 2))
        
        if self.game_over == X:
            x_win_image = self.large_font.render('X 승리', True, self.RED)
            self.screen.blit(x_win_image, x_win_image.get_rect(centerx=self.SCREEN_WIDTH // 2, centery=self.SCREEN_HEIGHT // 2))
        elif self.game_over == O:
            o_win_image = self.large_font.render('O 승리', True, self.RED)
            self.screen.blit(o_win_image, o_win_image.get_rect(centerx=self.SCREEN_WIDTH // 2, centery=self.SCREEN_HEIGHT // 2))
        elif self.game_over == DRAW:
            draw_image = self.large_font.render('무승부', True, self.RED)
            self.screen.blit(draw_image, draw_image.get_rect(centerx=self.SCREEN_WIDTH // 2, centery=self.SCREEN_HEIGHT // 2))
        
        pygame.display.update()
