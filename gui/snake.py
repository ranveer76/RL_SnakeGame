import pygame
from config import SNAKE

def draw_snake(surface, snake_body, cell_size, margin):
    for pos in snake_body:
        x, y = int(pos[0]), int(pos[1])
        rect = pygame.Rect(
            (x + margin // 2) * cell_size,
            (y + margin // 2) * cell_size,
            cell_size, cell_size
        )
        pygame.draw.rect(surface, SNAKE, rect)

