from board import *
from minimax import Minimax, AlphaBeta
# import ipdb
import time

humanPlayer, computerPlayer = W, B
# humanPlayer, computerPlayer = B, W

####################################
def playHuman():
    pygame.init()
    logo = pygame.image.load("logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("Reversi")
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    player = B
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
            print("No move. Pass")
            time.sleep(1)
            if nNoMove >= 2:
                game_over = board.winner()
                canvas.setGameOver(game_over)
                canvas.draw()
                # clock.tick(60)
            player = opponent(player)
            continue
        
        nNoMove = 0
        if game_over != 0:
            # print("game over")
            canvas.setGameOver(game_over)
            canvas.draw()
            continue

        if player == computerPlayer:
            # break
            print("Computer player searches..")
            # treeSearch = Minimax(board, 5, player, canvas)
            treeSearch = AlphaBeta(board, 6, player, canvas)
            (v, move) = treeSearch.run()
            # if move is None:
            #     ipdb.set_trace()
            print("Computer player selects", move[0]+1, move[1]+1, v)
        else:
            # print("drawValidMoves")
            # board.drawValidMoves(player)
            if move[0] < 0:
                continue
            if not board.isValidMove(player, move):
                print("Invalid move..")
                continue
        
        board.setMove(player, move)
        nStep += 1
        # if board.noValidMoves(player):
        #     game_over = board.winner()
        #     canvas.setGameOver(game_over)
        player = opponent(player)

        canvas.draw()
        clock.tick(60)

playHuman()    