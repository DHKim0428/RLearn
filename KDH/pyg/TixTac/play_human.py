from board import *
from minimax import Minimax

# humanPlayer, computerPlayer = X, O
humanPlayer, computerPlayer = O, X

####################################
def playHuman():
    pygame.init()
    logo = pygame.image.load("logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("Tix Tac Toe")
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    player = X
    nStep = 0

    canvas = Canvas(screen, SCREEN_WIDTH, SCREEN_HEIGHT)
    board = Board(canvas=canvas)
    game_over = 0

    canvas.draw()

    while True:
        move = canvas.get_events()
        if move == (-2, -2):
            pygame.quit()
            sys.exit()

        if game_over != 0:
            continue

        if player == computerPlayer:
            print("Computer player searches..")
            treeSearch = Minimax(board, 5, player, canvas)
            (v, move) = treeSearch.run()
            print("Computer player selects", move[0]+1, move[1]+1, v)
        else:
            if move[0] < 0:
                continue

            if not board.isValidMove(player, move):
                print("Invalid move..")
                continue
        
        board.setMove(player, move)
        print(canvas.grid)
        nStep += 1
        if board.noValidMoves(player):
            game_over = board.winner()
            canvas.setGameOver(game_over)
        player = opponent(player)

        canvas.draw()
        clock.tick(60)

playHuman()    