import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Window setup
WIDTH, HEIGHT = 600, 800
LANE_COUNT = 4
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🚗 Car Avoid Game - Enhanced Edition")

# Colors
ROAD_COLOR = (30, 30, 30)
LANE_COLOR = (80, 80, 80)
TEXT_COLOR = (255, 255, 255)
RED = (255, 60, 60)
BLUE = (0, 150, 255)
YELLOW = (255, 200, 60)
PURPLE = (180, 100, 255)
GRAY = (70, 70, 70)

# Clock and font
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 28, bold=True)
big_font = pygame.font.SysFont("Arial", 56, bold=True)

# Player setup
player_img = pygame.Surface((50, 90), pygame.SRCALPHA)
pygame.draw.rect(player_img, BLUE, (0, 0, 50, 90), border_radius=10)

player_x = WIDTH // 2 - 25
player_y = HEIGHT - 150
player_speed = 6
player_lives = 3

# NPC setup
npc_img = pygame.Surface((50, 90), pygame.SRCALPHA)
pygame.draw.rect(npc_img, RED, (0, 0, 50, 90), border_radius=10)

npc_list = []
for _ in range(4):
    lane = random.randint(0, LANE_COUNT - 1)
    npc_x = lane * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2) - 25
    npc_y = random.randint(-800, -100)
    npc_speed = random.uniform(4, 6)
    npc_list.append([npc_x, npc_y, npc_speed])

# Game variables
score = 0
level = 1
level_score = 0
lane_offset = 0
game_over = False

# --- Helper Functions ---


def draw_road():
    """Draw road and moving dashed lines."""
    screen.fill(ROAD_COLOR)
    lane_width = WIDTH // LANE_COUNT

    # Side borders
    pygame.draw.rect(screen, GRAY, (0, 0, 10, HEIGHT))
    pygame.draw.rect(screen, GRAY, (WIDTH - 10, 0, 10, HEIGHT))

    # Vertical lane lines
    for i in range(1, LANE_COUNT):
        pygame.draw.line(screen, LANE_COLOR, (i * lane_width, 0), (i * lane_width, HEIGHT), 3)

    # Moving dashed lines (center)
    global lane_offset
    lane_offset = (lane_offset + 5) % 40
    for y in range(-40, HEIGHT, 40):
        pygame.draw.line(screen, (255, 255, 255),
                         (lane_width * 2 - 2, y + lane_offset),
                         (lane_width * 2 - 2, y + 20 + lane_offset), 3)


def draw_hud():
    """Display player HUD (score, level, lives)."""
    score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
    level_text = font.render(f"Level: {level}", True, TEXT_COLOR)
    lives_text = font.render(f"Lives: {player_lives}", True, YELLOW)
    screen.blit(score_text, (20, 20))
    screen.blit(level_text, (WIDTH - 150, 20))
    screen.blit(lives_text, (20, 60))


def draw_game_over():
    """Display Game Over screen."""
    over_text = big_font.render("GAME OVER", True, RED)
    score_text = font.render(f"Final Score: {score}", True, TEXT_COLOR)
    restart_text = font.render("Press [R] to Restart", True, YELLOW)

    screen.blit(over_text, (WIDTH // 2 - 160, HEIGHT // 2 - 100))
    screen.blit(score_text, (WIDTH // 2 - 100, HEIGHT // 2 - 20))
    screen.blit(restart_text, (WIDTH // 2 - 160, HEIGHT // 2 + 40))


def reset_game():
    """Resets game state."""
    global player_x, player_lives, score, level, npc_list, game_over, level_score
    player_x = WIDTH // 2 - 25
    player_lives = 3
    score = 0
    level = 1
    level_score = 0
    game_over = False
    npc_list.clear()
    for _ in range(4):
        lane = random.randint(0, LANE_COUNT - 1)
        npc_x = lane * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2) - 25
        npc_y = random.randint(-800, -100)
        npc_speed = random.uniform(4, 6)
        npc_list.append([npc_x, npc_y, npc_speed])


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

        # Update NPC positions
        for npc in npc_list:
            npc[1] += npc[2]
            if npc[1] > HEIGHT:
                npc[1] = random.randint(-600, -100)
                npc[0] = random.randint(0, LANE_COUNT - 1) * (WIDTH // LANE_COUNT) + (WIDTH // LANE_COUNT // 2) - 25
                npc[2] = random.uniform(4 + level * 0.3, 6 + level * 0.5)
                score += 10
                level_score += 10

        # Level up every 200 points
        if level_score >= 200:
            level += 1
            level_score = 0

        # Collision detection
        player_rect = pygame.Rect(player_x, player_y, 50, 90)
        for npc in npc_list:
            npc_rect = pygame.Rect(npc[0], npc[1], 50, 90)
            if player_rect.colliderect(npc_rect):
                player_lives -= 1
                npc[1] = random.randint(-800, -100)
                if player_lives <= 0:
                    game_over = True
                break

        # Draw
        draw_road()
        screen.blit(player_img, (player_x, player_y))
        for npc in npc_list:
            screen.blit(npc_img, (npc[0], npc[1]))
        draw_hud()

    else:
        draw_game_over()
        if keys[pygame.K_r]:
            reset_game()

    pygame.display.flip()
