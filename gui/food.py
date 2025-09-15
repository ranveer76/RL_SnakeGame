import pygame
from config import FOOD

def draw_food(surface, food_pos, cell_size, margin, color=FOOD):
    x, y = int(food_pos[0]), int(food_pos[1])
    rect = pygame.Rect(
        (x + margin // 2) * cell_size,
        (y + margin // 2) * cell_size,
        cell_size, cell_size
    )
    pygame.draw.rect(surface, color, rect)
