import sys
import gym
import pygame
from game_2048_gym import Game2048Env

# Window Constants
GRID_SIZE = 4
CELL_PADDING = 5
WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE // (GRID_SIZE + 1)
FPS = 60

# Cell Colors 
BACKGROUND = (187, 173, 160)
EMPTY_CELL = (205, 193, 180)
CELL_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
}
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)

# Pygame setup globals
screen = None
clock = None

game = Game2048Env()

# setup pygame
def setup(GUI=True):
    global screen, clock
    if GUI:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("2048 Game")
        clock = pygame.time.Clock()

# Draw the gui
def draw_grid(game):
    screen.fill(BACKGROUND)

    title_font = pygame.font.Font(None, 60)
    score_font = pygame.font.Font(None, 36)

    title = title_font.render('2048', True, TEXT_DARK)
    screen.blit(title, (20, 20))

    score_text = score_font.render(f'Score: {game.score}', True, TEXT_DARK)
    screen.blit(score_text, (WINDOW_SIZE - 180, 30))

    # Draw grid
    x_offset = 20
    y_offset = 70
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            value = game.grid[i][j]
            x = j * CELL_SIZE + CELL_PADDING + x_offset
            y = i * CELL_SIZE + CELL_PADDING + y_offset

            # Draw cell background
            color = CELL_COLORS.get(value, CELL_COLORS[4096]) if value else EMPTY_CELL
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE - 2 * CELL_PADDING, 
                CELL_SIZE - 2 * CELL_PADDING), border_radius=5)

            # Draw number
            if value:
                font_size = 55 if value < 100 else (45 if value < 1000 else 35)
                font = pygame.font.Font(None, font_size)
                text_color = TEXT_DARK if value <= 4 else TEXT_LIGHT
                text = font.render(str(value), True, text_color)
                text_rect = text.get_rect(
                    center=(x + (CELL_SIZE - 2 * CELL_PADDING) // 2, y + (CELL_SIZE - 2 * CELL_PADDING) // 2))
                screen.blit(text, text_rect)

    # Print game over to screen
    if game.game_over:
        overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        overlay.set_alpha(180)
        overlay.fill((255, 255, 255))
        screen.blit(overlay, (0, 0))

        game_over_font = pygame.font.Font(None, 72)
        game_over_text = game_over_font.render('Game Over!', True, TEXT_DARK)
        screen.blit(game_over_text, (WINDOW_SIZE // 2 - game_over_text.get_width() // 2, 
            WINDOW_SIZE // 2 - 50))

    pygame.display.flip()


# Run the game visualized
def main():
    setup()  # initialize pygame
    game = Game2048Env()  # Use the Gym environment

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_LEFT:
                    game.step('LEFT')
                elif event.key == pygame.K_RIGHT:
                    game.step('RIGHT')
                elif event.key == pygame.K_UP:
                    game.step('UP')
                elif event.key == pygame.K_DOWN:
                    game.step('DOWN')

        draw_grid(game)
        clock.tick(FPS)


if __name__ == '__main__':
    main()