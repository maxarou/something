import numpy as np
import random

class UnfoldedEnvironment:
    def __init__(self, H, W, F, obstacle_prob=0.05, seed=None):
        self.H = H
        self.W = W
        self.F = F
        self.unfolded_W = W * F

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid = np.zeros((H, self.unfolded_W), dtype=int)

        # random obstacles
        mask = np.random.rand(H, self.unfolded_W) < obstacle_prob
        self.grid[mask] = 1

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.H and 0 <= gy < self.unfolded_W

    def is_obstacle(self, gx, gy):
        if not self.in_bounds(gx, gy):
            return True
        return self.grid[gx, gy] == 1

    def sample_free_cell(self):
        free = np.argwhere(self.grid == 0)
        idx = np.random.choice(len(free))
        return tuple(free[idx])
