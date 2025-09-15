import pygame, time, torch, os
from envs.game_env import SingleSnakeWrapper
from gui.renderer import Renderer
from config import CELL_SIZE, WINDOW_WIDTH as GRID_W, WINDOW_HEIGHT as GRID_H, FPS, MARGIN
from model.agent import DQNAgent
import tkinter as tk
from tkinter import filedialog, simpledialog
import io
import base64
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import threading
import queue
# import uvicorn
import asyncio
from starlette.websockets import WebSocketDisconnect
from PIL import Image

app = FastAPI()
frame_queue = queue.Queue(maxsize=2)
rl_stop_event = threading.Event()

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    rl_stop_event.clear()
    agent_thread = threading.Thread(target=main, daemon=True)
    if not any(t.name == "rl_agent_thread" for t in threading.enumerate()):
        agent_thread.name = "rl_agent_thread"
        agent_thread.start()
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1)
                await websocket.send_text(frame)
                await asyncio.sleep(1 / FPS)
            except queue.Empty:
                continue
    except WebSocketDisconnect:
        print("⚠️ WebSocket disconnected by client")
        rl_stop_event.set()

def show_preview(surface):
    preview = pygame.display.set_mode(surface.get_size())
    preview.blit(surface, (0, 0))
    pygame.display.flip()
    pygame.time.wait(5)

def get_frame_bytes(surface):
    size = surface.get_size()
    raw_bytes = pygame.image.tostring(surface, "RGB")
    img = Image.frombytes("RGB", size, raw_bytes)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")

def select_training_mode(window):
    font = pygame.font.SysFont(None, 40)
    options = [
        ("First Train", (100, 150, 200, 60)),
        ("Fine Tune",   (350, 150, 200, 60))
    ]
    buttons = [(text, pygame.Rect(*rect)) for text, rect in options]

    while True:
        window.fill((30, 30, 30))
        for text, rect in buttons:
            pygame.draw.rect(window, (70, 70, 200), rect)
            label = font.render(text, True, (255, 255, 255))
            window.blit(label, (rect.x + 20, rect.y + 10))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for text, rect in buttons:
                    if rect.collidepoint(event.pos):
                        return text.lower().replace(" ", "_")

def get_path_for_mode(mode):
    root = tk.Tk()
    root.withdraw()

    if mode == "first_train":
        folder = filedialog.askdirectory(title="Select folder to save new model")
        if not folder:
            return None
        filename = simpledialog.askstring("Filename", "Enter filename for new model (e.g. model.pth):")
        if not filename:
            filename = "model.pth"
        return os.path.join(folder, filename)

    elif mode == "fine_tune":
        return filedialog.askopenfilename(title="Select existing model file", filetypes=[("PyTorch Model", "*.pth")])

def get_file_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select model file",
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
    )
    return file_path

def select_mode(window):
    font = pygame.font.SysFont(None, 48)
    train_btn = pygame.Rect(100, 150, 200, 60)
    test_btn = pygame.Rect(400, 150, 200, 60)

    while True:
        window.fill((30, 30, 30))
        pygame.draw.rect(window, (0, 128, 0), train_btn)
        pygame.draw.rect(window, (0, 0, 128), test_btn)

        train_text = font.render("Train", True, (255, 255, 255))
        test_text = font.render("Test", True, (255, 255, 255))
        window.blit(train_text, (train_btn.x + 50, train_btn.y + 10))
        window.blit(test_text, (test_btn.x + 50, test_btn.y + 10))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if train_btn.collidepoint(event.pos):
                    return "train"
                elif test_btn.collidepoint(event.pos):
                    return "test"

def game():
    pygame.init()
    window = pygame.display.set_mode(
        ((GRID_W + MARGIN) * CELL_SIZE, (GRID_H + MARGIN) * CELL_SIZE)
    )
    pygame.display.set_caption("Snake RL")
    clock = pygame.time.Clock()

    mode = select_mode(window)

    if mode == "train":
        training_mode = select_training_mode(window)
        model_path = get_path_for_mode(training_mode)

        if model_path:
            from trainer.train import train_adaptive
            train_adaptive(model=model_path)
        else:
            print("⚠️ Training cancelled.")
        return
    else:
        model = get_file_path()
        agent = DQNAgent()
        agent.load(model)
        env = SingleSnakeWrapper()
        renderer = Renderer(window, GRID_W, GRID_H, CELL_SIZE, MARGIN)
        state = env.reset()
        score = 0
        length = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            with torch.no_grad():
                q_vals = agent.model(state.unsqueeze(0))
                action = torch.argmax(q_vals).item()

            next_state, reward, done = env.step(action)
            score += reward
            state = next_state

            snake_body = env.get_snake()
            food_pos = env.get_food()
            renderer.render(snake_body, food_pos)

            clock.tick(FPS)

            if done:
                print(f"Game Over. Final Length: {length}, Score: {score:.2f}")
                time.sleep(1)
                score = 0
                state = env.reset()
            else:
                length = len(snake_body)

def main():
    pygame.init()
    window = pygame.Surface(
        ((GRID_W + MARGIN) * CELL_SIZE, (GRID_H + MARGIN) * CELL_SIZE)
    )
    pygame.display.set_caption("Snake RL")
    clock = pygame.time.Clock()

    model = "models/model.pth"
    agent = DQNAgent()
    agent.load(model)
    env = SingleSnakeWrapper()
    renderer = Renderer(window, GRID_W, GRID_H, CELL_SIZE, MARGIN, headless=True)
    state = env.reset()
    score = 0
    length = 0
    while not rl_stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        with torch.no_grad():
            q_vals = agent.model(state.unsqueeze(0))
            action = torch.argmax(q_vals).item()
        next_state, reward, done = env.step(action)
        score += reward
        state = next_state
        snake_body = env.get_snake()
        food_pos = env.get_food()
        renderer.render(snake_body, food_pos)
        # show_preview(window)
        frame = get_frame_bytes(window)
        # with open("test_frame.png", "wb") as f:
        #     f.write(base64.b64decode(frame))
        frame_queue.put(frame)
        clock.tick(FPS)
        if done:
            print(f"Game Over. Final Length: {length}, Score: {score:.2f}")
            time.sleep(1)
            score = 0
            state = env.reset()
        else:
            length = len(snake_body)


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)