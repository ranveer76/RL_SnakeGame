import pygame
from gui.grid import draw_grid
from gui.snake import draw_snake
from gui.food import draw_food

class Renderer:
    def __init__(self, surface, grid_w, grid_h, cell_size=20, margin=2, headless=False):
        self.surface = surface
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.margin = margin
        self.headless = headless

    def render(self, snake_body, food_pos):
        self.surface.fill((255, 255, 255))
        draw_grid(self.surface, self.grid_w, self.grid_h, self.cell_size, self.margin)
        draw_snake(self.surface, snake_body, self.cell_size, self.margin)
        draw_food(self.surface, food_pos, self.cell_size, self.margin)
        if not self.headless:
            pygame.display.update()
