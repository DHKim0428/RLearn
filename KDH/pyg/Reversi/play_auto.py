from board import *
from minimax import Minimax, AlphaBeta

####################################
def playAuto():
    pygame.init()
    logo = pygame.image.load("logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("Reversi")
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    player = W
    nStep = 0
    nNoMove = 0

    canvas = Canvas(screen, SCREEN_WIDTH, SCREEN_HEIGHT)
    board = Board(canvas=canvas)
    game_over = 0

    canvas.draw()

    while True:
        move = canvas.get_events()
        if move == (-2, -2):
            pygame.quit()
            sys.exit()

        if board.noValidMoves(player):
            nNoMove += 1
            if nNoMove >= 2:
                game_over = board.winner()
                canvas.setGameOver(game_over)
                canvas.draw()
                # print("Done?")
                # print("Game Over?", game_over)
                # clock.tick(60)
            player = opponent(player)
            continue

        if game_over != 0:
            print("game over")
            canvas.setGameOver(game_over)
            canvas.draw()
            continue
        
        # print("Computer player searches..")
        # treeSearch = Minimax(board, 5, player, canvas)
        treeSearch = AlphaBeta(board, 5, player, canvas)
        (v, move) = treeSearch.run()
        # print("Computer player selects", move[0]+1, move[1]+1, v)

        board.setMove(player, move)
        # print(canvas.grid)
        nStep += 1
        # if board.noValidMoves(player):
        #     game_over = board.winner()
        #     canvas.setGameOver(game_over)
        player = opponent(player)

        canvas.draw()
        clock.tick(1)

playAuto()    