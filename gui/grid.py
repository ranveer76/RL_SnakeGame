import pygame
from config import BACKGROUND

def draw_grid(surface, grid_w, grid_h, cell_size, MARGIN,color=BACKGROUND):
    # for x in range(grid_w):
    #     for y in range(grid_h):
    #         rect = pygame.Rect(
    #             (x+MARGIN//2) * cell_size,
    #             (y+MARGIN//2) * cell_size,
    #             cell_size, cell_size
    #         )
    #         pygame.draw.rect(surface, color, rect, 1)
    rect = pygame.Rect(
        (MARGIN/2) * cell_size,
        (MARGIN/2) * cell_size,
        grid_w * cell_size,
        grid_h * cell_size
    )
    pygame.draw.rect(surface,color,rect)
