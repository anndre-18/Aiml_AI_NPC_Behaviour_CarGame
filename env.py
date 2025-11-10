# # env.py
# import numpy as np
# import random

# class CarAvoidEnv:
#     """
#     Minimal gym-like environment for car avoidance training.
#     Observation: [player_x_norm, npc_x_norm, npc_y_norm, npc_speed_norm, archetype_code]
#     Action: discrete 3 actions for NPC movement: [stay (0), move_left (1), move_right (2)]
#     The agent controls NPC behavior (i.e., learns how to move lane to collide with player).
#     """
#     def __init__(self, width=400, lanes=5, npc_count=3, archetype="neutral", max_steps=300):
#         self.width = width
#         self.lanes = lanes
#         self.lane_width = width / lanes
#         self.npc_count = npc_count
#         self.archetype = archetype  # "aggressive", "defensive", "neutral"
#         self.max_steps = max_steps

#         self.reset()

#     def reset(self):
#         self.player_lane = self.lanes // 2
#         # player tends to move left/right randomly with some bias
#         self.player_pos = np.array([self.player_lane, 0.9])  # lane, y (0 top,1 bottom)
#         # create NPCs as (lane, y)
#         self.npcs = []
#         for i in range(self.npc_count):
#             lane = random.randrange(self.lanes)
#             y = random.uniform(-1.0, -0.2) - i*0.3
#             speed = random.uniform(0.01, 0.03)
#             self.npcs.append([lane, y, speed])
#         self.steps = 0
#         self.score = 0
#         self.last_distance = self._min_distance()
#         return self._get_obs()

#     def step(self, action):
#         """
#         action: integer in [0..(3*npcs-1)] or we can assume agent controls single NPC index + move
#         For simplicity: action is (npc_index * 3) + move where move: 0 stay,1 left,2 right
#         """
#         self.steps += 1
#         npc_index = action // 3
#         move = action % 3

#         # clamp
#         npc_index = min(npc_index, self.npc_count - 1)
#         # apply move to that npc's lane
#         if move == 1:
#             self.npcs[npc_index][0] = max(0, self.npcs[npc_index][0] - 1)
#         elif move == 2:
#             self.npcs[npc_index][0] = min(self.lanes - 1, self.npcs[npc_index][0] + 1)

#         # Simulate player movement (stochastic, influenced by archetype)
#         # Player more likely to dodge where NPCs gather (makes environment dynamic)
#         if random.random() < 0.6:
#             # small random move
#             if random.random() < 0.5 and self.player_pos[0] > 0:
#                 self.player_pos[0] -= 1
#             elif self.player_pos[0] < self.lanes - 1:
#                 self.player_pos[0] += 1

#         # Update y for all NPCs (move toward player)
#         done = False
#         reward = 0.0
#         for npc in self.npcs:
#             y_before = npc[1]
#             npc[1] += npc[2]  # update y
#             # small lane drift depending on archetype to emulate style
#             if self.archetype == "aggressive" and random.random() < 0.2:
#                 # attempt to match player's lane
#                 if npc[0] < self.player_pos[0]:
#                     npc[0] = min(self.lanes - 1, npc[0] + 1)
#                 elif npc[0] > self.player_pos[0]:
#                     npc[0] = max(0, npc[0] - 1)
#             elif self.archetype == "defensive" and random.random() < 0.15:
#                 # avoid player's lane
#                 if npc[0] == self.player_pos[0]:
#                     npc[0] = max(0, npc[0] - 1) if random.random() < 0.5 else min(self.lanes - 1, npc[0] + 1)

#             # check collision if npc reaches bottom (y >= 1.0 normalized)
#             if npc[1] >= 1.0:
#                 if npc[0] == int(self.player_pos[0]):
#                     # collision -> high reward for NPC agent
#                     reward += 1.0
#                     self.score += 1
#                 else:
#                     # near miss: smaller reward (engagement)
#                     # measure horizontal distance
#                     dist = abs(npc[0] - self.player_pos[0])
#                     reward += max(0.0, 0.2 * (self.lanes - dist) / self.lanes)
#                 # respawn npc at top
#                 npc[1] = random.uniform(-1.0, -0.2)
#                 npc[0] = random.randrange(self.lanes)
#                 npc[2] = random.uniform(0.01, 0.04)

#         # reward shaping to balance challenge:
#         # encourage near-misses and occasional hits, penalize predictable spawn
#         # compute min distance to player among npcs
#         min_dist = self._min_distance()
#         # if distance decreased -> positive small reward
#         if min_dist < self.last_distance:
#             reward += 0.05
#         else:
#             reward -= 0.01
#         self.last_distance = min_dist

#         self.steps += 0
#         if self.steps >= self.max_steps:
#             done = True

#         obs = self._get_obs()
#         return obs, reward, done, {"score": self.score}

#     def _get_obs(self):
#         # Return flattened state: player's lane (one-hot), top-npc lanes & y positions
#         # To keep small, we return player lane normalized and for first npc: lane and y
#         # (This is a toy observation; you can expand)
#         npc0 = self.npcs[0]
#         archetype_code = 0
#         if self.archetype == "aggressive": archetype_code = 1
#         if self.archetype == "defensive": archetype_code = -1
#         return np.array([self.player_pos[0] / (self.lanes - 1),
#                          npc0[0] / (self.lanes - 1),
#                          npc0[1],
#                          npc0[2],
#                          archetype_code], dtype=np.float32)

#     def action_space(self):
#         return self.npc_count * 3

#     def observation_space_dim(self):
#         return 5

#     def _min_distance(self):
#         # Manhattan lane diff for nearest npc (lower is closer)
#         return min([abs(n[0] - self.player_pos[0]) + abs(n[1] - self.player_pos[1]) for n in self.npcs])









import numpy as np
import random

class CarAvoidEnv:
    """
    Minimal gym-like environment for car avoidance training.
    Observation: [player_x_norm, npc_x_norm, npc_y_norm, npc_speed_norm, archetype_code]
    Action: discrete 3 actions for NPC movement: [stay (0), move_left (1), move_right (2)]
    The agent controls NPC behavior (i.e., learns how to move lane to collide with player).
    """
    def __init__(self, width=400, lanes=5, npc_count=3, archetype="neutral", max_steps=300):
        self.width = width
        self.lanes = lanes
        self.lane_width = width / lanes
        self.npc_count = npc_count
        self.archetype = archetype  # "aggressive", "defensive", "neutral"
        self.max_steps = max_steps

        self.reset()

    def reset(self):
        self.player_lane = self.lanes // 2
        self.player_pos = np.array([self.player_lane, 0.9])  # lane, y (0 top, 1 bottom)
        self.npcs = []
        for i in range(self.npc_count):
            lane = random.randrange(self.lanes)
            y = random.uniform(-1.0, -0.2) - i * 0.3
            speed = random.uniform(0.01, 0.03)
            self.npcs.append([lane, y, speed])
        self.steps = 0
        self.score = 0
        self.last_distance = self._min_distance()
        return self._get_obs()

    def step(self, action):
        """
        action: integer in [0..(3*npcs-1)] or we can assume agent controls single NPC index + move
        For simplicity: action is (npc_index * 3) + move where move: 0 stay,1 left,2 right
        """
        self.steps += 1
        npc_index = action // 3
        move = action % 3

        npc_index = min(npc_index, self.npc_count - 1)
        if move == 1:
            self.npcs[npc_index][0] = max(0, self.npcs[npc_index][0] - 1)
        elif move == 2:
            self.npcs[npc_index][0] = min(self.lanes - 1, self.npcs[npc_index][0] + 1)

        # Player stochastic movement (adds unpredictability)
        if random.random() < 0.6:
            if random.random() < 0.5 and self.player_pos[0] > 0:
                self.player_pos[0] -= 1
            elif self.player_pos[0] < self.lanes - 1:
                self.player_pos[0] += 1

        done = False
        reward = 0.0

        # --- REWARD DESIGN IMPLEMENTATION ---
        min_dist = self._min_distance()

        if min_dist < self.last_distance:
            reward += 1.0   # NPC gets closer → +1
        elif 0.3 < min_dist <= 0.6:
            reward += 0.5   # Maintains moderate distance → +0.5
        elif min_dist > 0.6:
            reward -= 0.5   # Moves too far away → -0.5

        reward += 0.1  # Player survives → +0.1
        # --- END REWARD DESIGN ---

        # Update NPC behavior
        for npc in self.npcs:
            npc[1] += npc[2]  # Move downward

            # Archetype behavioral bias
            if self.archetype == "aggressive" and random.random() < 0.2:
                if npc[0] < self.player_pos[0]:
                    npc[0] = min(self.lanes - 1, npc[0] + 1)
                elif npc[0] > self.player_pos[0]:
                    npc[0] = max(0, npc[0] - 1)
            elif self.archetype == "defensive" and random.random() < 0.15:
                if npc[0] == self.player_pos[0]:
                    npc[0] += random.choice([-1, 1])
                    npc[0] = max(0, min(self.lanes - 1, npc[0]))

            # Check collision when NPC reaches bottom
            if npc[1] >= 1.0:
                if npc[0] == int(self.player_pos[0]):
                    reward -= 100.0  # Big penalty for collision
                    done = True
                # Respawn NPC
                npc[1] = random.uniform(-1.0, -0.2)
                npc[0] = random.randrange(self.lanes)
                npc[2] = random.uniform(0.01, 0.04)

        self.last_distance = min_dist

        if self.steps >= self.max_steps:
            done = True

        obs = self._get_obs()
        return obs, reward, done, {"score": self.score}

    def _get_obs(self):
        npc0 = self.npcs[0]
        archetype_code = 0
        if self.archetype == "aggressive": archetype_code = 1
        if self.archetype == "defensive": archetype_code = -1
        return np.array([self.player_pos[0] / (self.lanes - 1),
                         npc0[0] / (self.lanes - 1),
                         npc0[1],
                         npc0[2],
                         archetype_code], dtype=np.float32)

    def action_space(self):
        return self.npc_count * 3

    def observation_space_dim(self):
        return 5

    def _min_distance(self):
        return min([abs(n[0] - self.player_pos[0]) + abs(n[1] - self.player_pos[1]) for n in self.npcs])
