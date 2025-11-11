import pygame
import random
import sys
import numpy as np

# --- Window setup ---
pygame.init()
WIDTH, HEIGHT = 400, 600
LANE_COUNT = 4
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🚗 Car Avoid Game - Player Attack NPC")
clock = pygame.time.Clock()

# --- Colors ---
ROAD_COLOR = (40, 40, 40)
LANE_COLOR = (100, 100, 100)
TEXT_COLOR = (255, 255, 255)
RED = (255, 60, 60)
BLUE = (0, 150, 255)
YELLOW = (255, 200, 60)
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
OVERLAY = (0, 0, 0, 180)

# --- Fonts ---
font = pygame.font.SysFont("Arial", 26, bold=True)
big_font = pygame.font.SysFont("Arial", 56, bold=True)

# --- Player setup ---
player_img = pygame.image.load("assets/player_car.png").convert_alpha()
player_img = pygame.transform.scale(player_img, (50, 90))
player_x = WIDTH // 2 - 25
player_y = HEIGHT - 150
player_speed = 6
player_lives = 3

# --- Coins (defensive NPC) ---
coin_img = pygame.image.load("assets/coins.png").convert_alpha()
coin_img = pygame.transform.scale(coin_img, (50, 90))
coins = []
for _ in range(4):
    lane = random.randint(0, LANE_COUNT - 1)
    coin_x = lane * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2 - 25)
    coin_y = random.randint(-800, -100)
    coin_speed = random.uniform(4, 6)
    coins.append([coin_x, coin_y, coin_speed])

# --- Enemy NPCs (attacking) ---
enemy_img = pygame.image.load("assets/enemy_car.png").convert_alpha()
enemy_img = pygame.transform.scale(enemy_img, (50, 90))
enemies = []
for _ in range(3):
    lane = random.randint(0, LANE_COUNT - 1)
    enemy_x = lane * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2 - 25)
    enemy_y = random.randint(-800, -100)
    enemy_speed = random.uniform(5, 7)
    enemies.append([enemy_x, enemy_y, enemy_speed])

# --- Game variables ---
score = 0
level = 1
lane_offset = 0
game_over = False

# --- Q-Learning Setup for coins ---
actions = ["move_left", "move_right", "accelerate", "decelerate"]
alpha = 0.1
gamma = 0.9
epsilon = 0.2
Q = {}

def get_state(player_x, npc_x):
    player_lane = player_x // (WIDTH // LANE_COUNT)
    npc_lane = npc_x // (WIDTH // LANE_COUNT)
    return (player_lane, npc_lane)

def choose_action(state):
    if state not in Q:
        Q[state] = [0] * len(actions)
    if random.random() < epsilon:
        return random.choice(range(len(actions)))
    return np.argmax(Q[state])

def update_Q(state, action_idx, reward, next_state):
    if next_state not in Q:
        Q[next_state] = [0] * len(actions)
    Q[state][action_idx] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action_idx])

def perform_npc_action(npc, action_idx):
    action = actions[action_idx]
    lane_width = WIDTH // LANE_COUNT
    move_step = 5
    if action == "move_left":
        npc[0] -= move_step
        if npc[0] < 0: npc[0] = 0
    elif action == "move_right":
        npc[0] += move_step
        if npc[0] > WIDTH - 50: npc[0] = WIDTH - 50
    elif action == "accelerate":
        npc[1] += min(npc[2] + 3, HEIGHT - 50)
    elif action == "decelerate":
        npc[1] += max(1, npc[2] - 2)
    else:
        npc[1] += npc[2]
    npc[0] = int(npc[0])
    npc[1] = int(npc[1])

# --- Helper Functions ---
def draw_road():
    global lane_offset
    screen.fill((30, 30, 30))
    lane_width = WIDTH // LANE_COUNT
    for y in range(HEIGHT):
        color_val = max(0, min(ROAD_COLOR[0] + y // 20, 60))
        pygame.draw.line(screen, (color_val, color_val, color_val), (0, y), (WIDTH, y))
    pygame.draw.rect(screen, GRAY, (0, 0, 12, HEIGHT))
    pygame.draw.rect(screen, GRAY, (WIDTH - 12, 0, 12, HEIGHT))
    for i in range(1, LANE_COUNT):
        pygame.draw.line(screen, LANE_COLOR, (i * lane_width, 0), (i * lane_width, HEIGHT), 3)
    lane_offset = (lane_offset + 8) % 40
    for y in range(-40, HEIGHT, 40):
        pygame.draw.line(screen, WHITE,
                         (lane_width * 2 - 2, y + lane_offset),
                         (lane_width * 2 - 2, y + 20 + lane_offset), 4)

def draw_hud():
    panel_width = 150
    panel_height = 40
    pygame.draw.rect(screen, GRAY, (15, 15, panel_width, panel_height), border_radius=8)
    pygame.draw.rect(screen, GRAY, (WIDTH - panel_width - 15, 15, panel_width, panel_height), border_radius=8)
    score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
    level_text = font.render(f"Level: {level}", True, TEXT_COLOR)
    lives_text = font.render(f"Lives: {player_lives}", True, YELLOW)
    screen.blit(score_text, (25, 20))
    screen.blit(level_text, (WIDTH - panel_width, 20))
    screen.blit(lives_text, (25, 60))

def draw_game_over():
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill(OVERLAY)
    screen.blit(overlay, (0, 0))
    over_text = big_font.render("GAME OVER", True, RED)
    score_text = font.render(f"Final Score: {score}", True, WHITE)
    restart_text = font.render("Press [R] to Restart", True, YELLOW)
    screen.blit(over_text, (WIDTH // 2 - over_text.get_width() // 2, HEIGHT // 2 - 100))
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 20))
    screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 40))

def reset_game():
    global player_x, player_lives, score, level, coins, enemies, game_over
    player_x = WIDTH // 2 - 25
    player_lives = 3
    score = 0
    level = 1
    game_over = False
    coins.clear()
    enemies.clear()
    for _ in range(4):
        lane = random.randint(0, LANE_COUNT - 1)
        coin_x = lane * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2 - 25)
        coin_y = random.randint(-800, -100)
        coin_speed = random.uniform(4, 6)
        coins.append([coin_x, coin_y, coin_speed])
    for _ in range(3):
        lane = random.randint(0, LANE_COUNT - 1)
        enemy_x = lane * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2 - 25)
        enemy_y = random.randint(-800, -100)
        enemy_speed = random.uniform(5, 7)
        enemies.append([enemy_x, enemy_y, enemy_speed])

# --- Main Game Loop ---
running = True
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if not game_over:
        # Player movement
        if keys[pygame.K_LEFT] and player_x > 20:
            player_x -= player_speed
        if keys[pygame.K_RIGHT] and player_x < WIDTH - 70:
            player_x += player_speed

        player_rect = pygame.Rect(player_x, player_y, 50, 90)

        # Move coins (defensive NPCs) with Q-Learning
        for coin in coins:
            state = get_state(player_x, coin[0])
            action_idx = choose_action(state)
            perform_npc_action(coin, action_idx)
            reward = 0
            if abs(player_x - coin[0]) > 50:
                reward += 1
            coin_rect = pygame.Rect(coin[0], coin[1], 50, 90)
            if coin_rect.colliderect(player_rect):
                reward -= 1
            next_state = get_state(player_x, coin[0])
            update_Q(state, action_idx, reward, next_state)
            if coin[1] > HEIGHT:
                coin[1] = random.randint(-600, -100)
                coin[0] = random.randint(0, LANE_COUNT - 1) * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2 - 25)
                coin[2] = random.uniform(4 + level*0.3, 6 + level*0.5)

        # Player collects coins → score +1
        for coin in coins:
            coin_rect = pygame.Rect(coin[0], coin[1], 50, 90)
            if player_rect.colliderect(coin_rect):
                score += 1
                coin[1] = random.randint(-800, -100)
                coin[0] = random.randint(0, LANE_COUNT - 1) * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2 - 25)
                coin[2] = random.uniform(4 + level*0.3, 6 + level*0.5)
                break

        # Move enemies (attacking NPCs)
        for enemy in enemies:
            enemy[1] += enemy[2]
            if enemy[1] > HEIGHT:
                enemy[1] = random.randint(-800, -100)
                enemy[0] = random.randint(0, LANE_COUNT - 1) * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2 - 25)
                enemy[2] = random.uniform(5, 7)
            enemy_rect = pygame.Rect(enemy[0], enemy[1], 50, 90)
            if player_rect.colliderect(enemy_rect):
                player_lives -= 1
                enemy[1] = random.randint(-800, -100)
                if player_lives <= 0:
                    game_over = True

        # Level up every 20 points
        if score >= level * 20:
            level += 1

        # Draw
        draw_road()
        screen.blit(player_img, (player_x, player_y))
        for coin in coins:
            screen.blit(coin_img, (coin[0], coin[1]))
        for enemy in enemies:
            screen.blit(enemy_img, (enemy[0], enemy[1]))
        draw_hud()

    else:
        draw_game_over()
        if keys[pygame.K_r]:
            reset_game()

    pygame.display.flip()
