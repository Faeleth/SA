import pygame
import random
import threading
import time
from voice_control import listen_command, set_silence_threshold

# --- konfiguracja ---
WIDTH, HEIGHT = 600, 400
CELL = 20
FPS = 2

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake – Voice Control")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
big_font = pygame.font.SysFont(None, 60)

# --- stan gry ---
snake = [(100, 100), (80, 100), (60, 100)]
direction = "RIGHT"
next_direction = "RIGHT"
food = (random.randrange(0, WIDTH, CELL), random.randrange(0, HEIGHT, CELL))
score = 0
running = True
game_active = True
command_buffer = None
last_command_time = 0

# --- reset gry ---
def reset_game():
    global snake, direction, next_direction, food, score, game_active
    snake = [(100, 100), (80, 100), (60, 100)]
    direction = "RIGHT"
    next_direction = "RIGHT"
    food = (random.randrange(0, WIDTH, CELL), random.randrange(0, HEIGHT, CELL))
    score = 0
    game_active = True

# --- watek rozpoznawania mowy ---
def voice_thread():
    global command_buffer, last_command_time
    while running:
        try:
            cmd = listen_command()
            if cmd:
                command_buffer = cmd
                last_command_time = time.time()
        except Exception as e:
            print(f"Voice error: {e}")
            pass

# Start watku rozpoznawania mowy
voice_thread_obj = threading.Thread(target=voice_thread, daemon=True)
voice_thread_obj.start()

# --- funkcje rysowania ---
def draw_snake():
    for i, segment in enumerate(snake):
        color = GREEN if i == 0 else (0, 200, 0)  # Glowa jasniejsza
        pygame.draw.rect(screen, color, (*segment, CELL, CELL))
        pygame.draw.rect(screen, BLACK, (*segment, CELL, CELL), 1)

def draw_food():
    pygame.draw.rect(screen, RED, (*food, CELL, CELL))
    pygame.draw.circle(screen, YELLOW, (food[0] + CELL//2, food[1] + CELL//2), 3)

def draw_score():
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (10, 10))

def game_over_screen():
    screen.fill(BLACK)
    
    game_over_text = big_font.render("GAME OVER", True, RED)
    score_text = font.render(f"Final Score: {score}", True, WHITE)
    restart_text = font.render("Speak 'gora' to restart or wait 3 seconds", True, YELLOW)
    
    screen.blit(game_over_text, (WIDTH // 2 - 200, HEIGHT // 2 - 100))
    screen.blit(score_text, (WIDTH // 2 - 120, HEIGHT // 2))
    screen.blit(restart_text, (WIDTH // 2 - 250, HEIGHT // 2 + 80))
    
    pygame.display.flip()
    
    # Wait for restart or timeout
    start_time = time.time()
    while time.time() - start_time < 3:
        if command_buffer == "gora":
            return True  # restart
        pygame.time.delay(100)
    
    return False  # restart

# --- glowna petla gry ---
print("=" * 50)
print("🎤 Snake Game - Voice Control")
print("=" * 50)
print("Commands: gora (UP), dol (DOWN), lewo (LEFT), prawo (RIGHT)")
print("Calibrating microphone...\n")

set_silence_threshold()
print("Ready! Start speaking commands...\n")

game_restart = True

while running:
    if game_restart:
        reset_game()
        game_restart = False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    
    if not game_active:
        # Ekran konca gry
        if game_over_screen():
            game_restart = True
        else:
            game_restart = True
        continue
    
    # Przetworz komende glosowa
    if command_buffer:
        cmd = command_buffer.lower()
        command_buffer = None
        
        # Blokuj przeciwny ruch
        if cmd == "gora" and direction != "DOWN":
            next_direction = "UP"
        elif cmd == "dol" and direction != "UP":
            next_direction = "DOWN"
        elif cmd == "lewo" and direction != "RIGHT":
            next_direction = "LEFT"
        elif cmd == "prawo" and direction != "LEFT":
            next_direction = "RIGHT"
    
    # Aktualizacja kierunku
    direction = next_direction
    
    # Przesun weza
    x, y = snake[0]
    if direction == "UP":
        y -= CELL
    elif direction == "DOWN":
        y += CELL
    elif direction == "LEFT":
        x -= CELL
    elif direction == "RIGHT":
        x += CELL
    
    new_head = (x, y)
    
    # Sprawdz kolizje
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or new_head in snake:
        game_active = False
        continue
    
    # Dodaj nowa glowe
    snake.insert(0, new_head)
    
    # Sprawdz czy zjedzono jedzenie
    if new_head == food:
        score += 1
        food = (random.randrange(0, WIDTH, CELL), random.randrange(0, HEIGHT, CELL))
    else:
        snake.pop()
    
    # Narysuj wszystko
    screen.fill(BLACK)
    draw_snake()
    draw_food()
    draw_score()
    pygame.display.flip()
    
    clock.tick(FPS)

pygame.quit()
print("Goodbye!")
