import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import time
from env import AddPad
from episode_runner import run_episode
from oracle import Oracle
from random_agent import RandomAgent
from env import Cursor, EMPTY

do_sleep = False

pygame.init()
screen = pygame.display.set_mode((640, 360))
clock = pygame.time.Clock()
running = True

icon = pygame.image.load("icon.png")
pygame.display.set_icon(icon)
pygame.display.set_caption("AddPad")

font_digit = pygame.font.SysFont("consolas", 42)
font_carry = pygame.font.SysFont("consolas", 18)

env = AddPad(max_digits=3)
policy = Oracle()

obs = env.reset()
last_action = None
last_reward = 0.0
last_info = {}
dt = 0.0

center = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    screen.fill("beige")

    action = policy.act(obs)
    obs, reward, done, info = env.step(action)
    last_action = action
    last_reward = reward
    last_info = info

    A = obs["A"]
    B = obs["B"]
    cursor_val = obs["cursor"]
    pad = obs["pad"]

    def text(s, x, y, color=(20, 20, 20), font=font_digit):
        surf = font.render(s, True, color)
        rect = surf.get_rect(center=(x, y))
        screen.blit(surf, rect)

    def digit(n, place):
        ans = (n // place) % 10
        if ans == 0 and n // (place + 1) == 0:
            if place == 1:
                return 0
            return ""
        return ans
    
    def slot_center(i):
        cx = screen.get_width() / 2
        cy = screen.get_height() / 2

        if i % 2 == 0:
            col = i // 2
            x = cx + 50 - (50 * col)
            y = cy + 100
        else:
            x = cx + 5 - (50 * i / 2)
            y = cy - 25
        return x, y
    
    def highlight_answer(correct):
        cx = screen.get_width() / 2
        cy = screen.get_height() / 2
        answer_y = cy + 100
        
        left_x = cx + 50 - (50 * 3)
        right_x = cx + 50
        
        tw, th = font_digit.size("8")
        box_width = right_x - left_x + tw + 20
        box_height = th + 24
        box_center_x = (left_x + right_x) / 2
        
        if correct:
            fill_color = (0, 200, 0, 120)
            border_color = (0, 150, 0)
        else:
            fill_color = (200, 0, 0, 120)
            border_color = (150, 0, 0)
        
        draw_highlight((box_center_x, answer_y), box_width, box_height, fill_rgba=fill_color, border_rgb=border_color)

    def draw_highlight(center_xy, w, h, fill_rgba=(255, 240, 120, 120), border_rgb=(40, 40, 40), border_px=1):
        x, y = center_xy
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (x, y)

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill(fill_rgba)
        screen.blit(overlay, rect.topleft)

        pygame.draw.rect(screen, border_rgb, rect, border_px, border_radius=6)

    text(str(digit(A, 100)), screen.get_width() / 2 - 50, screen.get_height() / 2)
    text(str(digit(A, 10)), screen.get_width() / 2, screen.get_height() / 2)
    text(str(digit(A, 1)), screen.get_width() / 2 + 50, screen.get_height() / 2)

    text(str(digit(B, 100)), screen.get_width() / 2 - 50, (screen.get_height() / 2) + 50)
    text(str(digit(B, 10)), screen.get_width() / 2, (screen.get_height() / 2) + 50)
    text(str(digit(B, 1)), screen.get_width() / 2 + 50, (screen.get_height() / 2) + 50)

    text("+", screen.get_width() / 2 - 100, (screen.get_height() / 2) + 50)

    pygame.draw.line(screen, 
                     "black", 
                     pygame.Vector2(screen.get_width() / 2 - 100, screen.get_height() / 2 + 75), 
                     pygame.Vector2(screen.get_width() / 2 + 75, screen.get_height() / 2 + 75)
                    )
    
    for i, v in enumerate(pad):
        if v != EMPTY:
            x, y = slot_center(i)
            if i % 2 == 0:
                # digit
                text(str(v), x, y)
            else:
                # carry
                text(str(v), x, y, font=font_carry)

    x, y = slot_center(cursor_val - 1)
    if (cursor_val - 1) % 2 == 0:
        # digit
        tw, th = font_digit.size("8")
        pad_x, pad_y = 18, 12
    else:
        # carry
        tw, th = font_carry.size("8")
        pad_x, pad_y = 10, 8

    if not done:
        if not info.get("illegal"):
            draw_highlight((x, y), tw + pad_x, th + pad_y)
        else:
            draw_highlight((x, y), tw + pad_x, th + pad_y, fill_rgba=(200, 0, 0, 120))

    if done: highlight_answer(env.is_correct())
    if do_sleep: time.sleep(0.1)

    pygame.display.flip()

    if done:
        if do_sleep: time.sleep(3)
        obs = env.reset()

    dt = clock.tick(60) / 1000

pygame.quit()