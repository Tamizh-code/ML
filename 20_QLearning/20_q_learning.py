# Experiment 20: Q-learning in a small 4x4 GridWorld
import numpy as np
import random

random.seed(42)
np.random.seed(42)

SIZE = 4
START, GOAL = (0, 0), (3, 3)
OBSTACLES = {(1, 1), (2, 1)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
Q = np.zeros((SIZE * SIZE, len(ACTIONS)))

def state_id(pos):
    return pos[0] * SIZE + pos[1]

def step(pos, action):
    dr, dc = ACTIONS[action]
    nxt = (max(0, min(SIZE - 1, pos[0] + dr)),
           max(0, min(SIZE - 1, pos[1] + dc)))
    if nxt in OBSTACLES:
        nxt = pos
    reward = 10 if nxt == GOAL else -1
    return nxt, reward, nxt == GOAL

alpha, gamma, epsilon = 0.1, 0.9, 0.15
for episode in range(1500):
    pos = START
    for _ in range(50):
        s = state_id(pos)
        action = random.randrange(4) if random.random() < epsilon else int(np.argmax(Q[s]))
        nxt, reward, done = step(pos, action)
        ns = state_id(nxt)
        Q[s, action] += alpha * (reward + gamma * np.max(Q[ns]) - Q[s, action])
        pos = nxt
        if done:
            break

print("Learned Q-table:\n", np.round(Q, 2))
pos, path = START, [START]
for _ in range(20):
    action = int(np.argmax(Q[state_id(pos)]))
    pos, _, done = step(pos, action)
    path.append(pos)
    if done:
        break
print("Greedy path from start:", path)
