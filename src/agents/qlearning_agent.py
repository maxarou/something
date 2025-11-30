# src/agents/qlearning_agent.py
import numpy as np
import random
import pickle
from collections import defaultdict

ACTIONS = ['up', 'down', 'left', 'right', 'floor_up', 'floor_down']

class QLearningAgent:
    def __init__(self, agent_id, env, start_pos,
                 view_radius=1, alpha=0.5, gamma=0.95, epsilon=0.3):
        self.id = agent_id
        self.env = env
        self.pos = start_pos  # tuple (gx,gy)
        self.view_radius = view_radius
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q: dict state -> np.array(len(ACTIONS))
        self.Q = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=float))
        # local visited (boolean map same shape as env.grid)
        self.local_visited = np.zeros_like(env.grid, dtype=bool)
        gx, gy = start_pos
        self.local_visited[int(gx), int(gy)] = True

    # --- state representation: local patch flattened tuple ---
    def observe(self, pos=None):
        if pos is None:
            pos = self.pos
        gx, gy = int(pos[0]), int(pos[1])
        r = self.view_radius
        patch = []
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = gx + dx, gy + dy
                if not self.env.in_bounds(nx, ny):
                    patch.append(-1)            # out-of-bounds / obstacle-like
                elif self.env.is_obstacle(nx, ny):
                    patch.append(-1)
                elif self.local_visited[nx, ny]:
                    patch.append(1)             # visited
                else:
                    patch.append(0)             # unseen free
        return tuple(patch)

    # epsilon-greedy action selection
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(len(ACTIONS))
        qvals = self.Q[state]
        maxv = np.max(qvals)
        best = [i for i, v in enumerate(qvals) if v == maxv]
        return random.choice(best)

    # propose move target (global coords) and invalid flag
    def propose_move(self, action_idx):
        gx, gy = int(self.pos[0]), int(self.pos[1])
        W = self.env.W
        act = ACTIONS[action_idx]
        if act == 'up':
            tx, ty = gx-1, gy
        elif act == 'down':
            tx, ty = gx+1, gy
        elif act == 'left':
            tx, ty = gx, gy-1
        elif act == 'right':
            tx, ty = gx, gy+1
        elif act == 'floor_up':
            tx, ty = gx, gy + W
        elif act == 'floor_down':
            tx, ty = gx, gy - W
        else:
            tx, ty = gx, gy

        invalid = (not self.env.in_bounds(tx, ty)) or self.env.is_obstacle(tx, ty)
        return (int(tx), int(ty)), invalid

    def q_update(self, s, a, reward, s_next):
        q = self.Q[s][a]
        q_next = np.max(self.Q[s_next]) if s_next is not None else 0.0
        self.Q[s][a] = q + self.alpha * (reward + self.gamma * q_next - q)

    # persist/load Q-table to bytes (pickle)
    def get_q_bytes(self):
        return pickle.dumps(dict(self.Q))

    def load_q_bytes(self, b):
        data = pickle.loads(b)
        self.Q = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=float), data)
