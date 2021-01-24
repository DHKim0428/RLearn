import pygame
import sys

def is_free_position(grid, column_index, row_index):
    if column_index < 0 or column_index > 2:
        return False
    if row_index < 0 or column_index > 2:
        return False
    return grid[row_index][column_index] == ' '

def is_winner(grid_row_sum, grid_column_sum, grid_skew_sum, mark):
    sign = 1 if mark == 'X' else 4
    for s in grid_row_sum:
        if s == sign * 3: return True
    for s in grid_column_sum: 
        if s == sign * 3: return True
    for s in grid_skew_sum:
        if s == sign * 3: return True
    return False

def is_grid_full(grid_row_sum):
    for s in grid_row_sum:
        if s == 0: return False
    return True


pygame.init()
logo = pygame.image.load("logo32x32.png")
pygame.display.set_icon(logo)
pygame.display.set_caption("Tix Tac Toe")
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
large_font = pygame.font.SysFont('Noto Sans CJK HK', 72)
CELL_SIZE = 200
COLUMN_COUNT = 3
ROW_COUNT = 3
X_WIN = 1
O_WIN = 2
DRAW = 3
game_over = 0

grid = [[' '] * COLUMN_COUNT for i in range(ROW_COUNT)]
grid_row_sum = [0] * ROW_COUNT
grid_column_sum = [0] * COLUMN_COUNT
grid_skew_sum = [0, 0]
turn = 0
count = 0

while True:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and game_over == 0:
            column_index = event.pos[0] // CELL_SIZE
            row_index = event.pos[1] // CELL_SIZE
            print(column_index, row_index)
            if not is_free_position(grid, column_index, row_index):
                continue
            if turn == 0:
                grid[row_index][column_index] = 'X'
                grid_row_sum[row_index] += 1
                grid_column_sum[column_index] += 1
                grid_skew_sum[0] += 1 if row_index == column_index else 0
                grid_skew_sum[1] += 1 if row_index + column_index + 1 == COLUMN_COUNT else 0
                count += 1

                if is_winner(grid_row_sum, grid_column_sum, grid_skew_sum, 'X'):
                    game_over = X_WIN
                elif count >= 9:
                    game_over = DRAW

                turn = 1
            elif turn == 1:
                grid[row_index][column_index] = 'O'
                grid_row_sum[row_index] += 4
                grid_column_sum[column_index] += 4
                grid_skew_sum[0] += 4 if row_index == column_index else 0
                grid_skew_sum[1] += 4 if row_index + column_index + 1 == COLUMN_COUNT else 0
                count += 1

                if is_winner(grid_row_sum, grid_column_sum, grid_skew_sum, 'O'):
                    game_over = O_WIN
                elif count >= 9:
                    game_over = DRAW

                turn = 0

    for column_index in range(COLUMN_COUNT):
        for row_index in range(ROW_COUNT):
            pygame.draw.rect(screen, WHITE, pygame.Rect(column_index * CELL_SIZE, row_index * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
            mark = grid[row_index][column_index]
            if mark == 'X':
                X_image = large_font.render('X', True, YELLOW)
                screen.blit(X_image, X_image.get_rect(centerx=column_index * CELL_SIZE + CELL_SIZE // 2, 
                                                                                                centery=row_index * CELL_SIZE + CELL_SIZE // 2))
            elif mark == 'O':
                O_image = large_font.render('O', True, WHITE)
                screen.blit(O_image, O_image.get_rect(centerx=column_index * CELL_SIZE + CELL_SIZE // 2, 
                                                                                                centery=row_index * CELL_SIZE + CELL_SIZE // 2))
    
    if game_over == X_WIN:
        x_win_image = large_font.render('X 승리', True, RED)
        screen.blit(x_win_image, x_win_image.get_rect(centerx=SCREEN_WIDTH // 2, centery=SCREEN_HEIGHT // 2))
    elif game_over == O_WIN:
        o_win_image = large_font.render('O 승리', True, RED)
        screen.blit(o_win_image, o_win_image.get_rect(centerx=SCREEN_WIDTH // 2, centery=SCREEN_HEIGHT // 2))
    elif game_over == DRAW:
        draw_image = large_font.render('무승부', True, RED)
        screen.blit(draw_image, draw_image.get_rect(centerx=SCREEN_WIDTH // 2, centery=SCREEN_HEIGHT // 2))
        

    pygame.display.update()
    clock.tick(60)

