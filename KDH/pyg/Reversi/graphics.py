from constant import *
import pygame
import sys

class Canvas:
    def __init__(self, screen, width=600, height=600):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (27, 150, 64)
        self.DARKGREEN = (28, 77, 42)
        self.RED = (255, 0, 0)
        self.large_font = pygame.font.SysFont('Noto Sans CJK HK', 72)
        self.CELL_SIZE = 100
        self.screen = screen
        self.SCREEN_WIDTH = width
        self.SCREEN_HEIGHT = height

        self.grid = [[EMPTY] * N for i in range(N)]
        self.grid[N//2 - 1][N//2 - 1] = W
        self.grid[N//2 - 1][N//2] = B
        self.grid[N//2][N//2 - 1] = B
        self.grid[N//2][N//2] = W
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
    
    def addMark(self, grid):
        self.grid = grid
        self.turn += 1
    
    def setGameOver(self, result):
        self.game_over = result

    def draw(self):
        self.screen.fill(self.GREEN)
        for row in range(N):
            for col in range(N):
                pygame.draw.rect(self.screen, self.DARKGREEN, pygame.Rect(col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), 1)
                mark = self.grid[row][col]
                if mark == B:
                    pygame.draw.circle(self.screen, self.BLACK, 
                                        (col * self.CELL_SIZE + self.CELL_SIZE // 2, row * self.CELL_SIZE + self.CELL_SIZE //2), 
                                        self.CELL_SIZE // 2)
                elif mark == W:
                    pygame.draw.circle(self.screen, self.WHITE, 
                                        (col * self.CELL_SIZE + self.CELL_SIZE // 2, row * self.CELL_SIZE + self.CELL_SIZE //2), 
                                        self.CELL_SIZE // 2)
                elif mark == VALID:
                    pygame.draw.circle(self.screen, self.RED, 
                                        (col * self.CELL_SIZE + self.CELL_SIZE // 2, row * self.CELL_SIZE + self.CELL_SIZE //2), 
                                        self.CELL_SIZE // 6)
        
        if self.game_over == B:
            b_win_image = self.large_font.render('흑 승리', True, self.RED)
            self.screen.blit(b_win_image, b_win_image.get_rect(centerx=self.SCREEN_WIDTH // 2, centery=self.SCREEN_HEIGHT // 2))
        elif self.game_over == W:
            w_win_image = self.large_font.render('백 승리', True, self.RED)
            self.screen.blit(w_win_image, w_win_image.get_rect(centerx=self.SCREEN_WIDTH // 2, centery=self.SCREEN_HEIGHT // 2))
        elif self.game_over == DRAW:
            draw_image = self.large_font.render('무승부', True, self.RED)
            self.screen.blit(draw_image, draw_image.get_rect(centerx=self.SCREEN_WIDTH // 2, centery=self.SCREEN_HEIGHT // 2))
        
        pygame.display.update()
