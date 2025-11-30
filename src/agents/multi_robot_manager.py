# src/agents/multi_robot_manager.py
import numpy as np
import random
from collections import defaultdict
from .qlearning_agent import QLearningAgent, ACTIONS
from src.storage.mongo_client import MongoStorage  # existing module

class MultiRobotExplorer:
    def __init__(self, env, num_robots=2, sync_K=5, view_radius=1, mongo_log=True, seed=None):
        self.env = env
        self.num_robots = num_robots
        self.sync_K = sync_K
        self.view_radius = view_radius
        self.global_visited = np.zeros_like(env.grid, dtype=bool)
        self.robots = []
        self.step_count = 0
        self.episode = 0
        self.storage = MongoStorage() if mongo_log else None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._spawn_robots()

    def _spawn_robots(self):
        placed = set()
        for rid in range(self.num_robots):
            for _ in range(500):
                gx, gy = self.env.sample_free_cell()
                if (int(gx), int(gy)) in placed:
                    continue
                placed.add((int(gx), int(gy)))
                agent = QLearningAgent(rid, self.env, (int(gx), int(gy)), view_radius=self.view_radius)
                self.robots.append(agent)
                self.global_visited[int(gx), int(gy)] = True
                if self.storage:
                    self.storage.log_visit(rid, int(gx), int(gy), step=0, floor=(gy // self.env.W))
                break

    def reset_for_episode(self):
        # optional: re-sample robots or keep positions
        self.step_count = 0
        self.episode += 1
        # clear global visited but keep obstacles
        self.global_visited = np.zeros_like(self.env.grid, dtype=bool)
        # reinitialize robots local visited and positions (keep same starting pos)
        for r in self.robots:
            r.local_visited = np.zeros_like(self.env.grid, dtype=bool)
            gx, gy = r.pos
            r.local_visited[int(gx), int(gy)] = True
            self.global_visited[int(gx), int(gy)] = True

    def step(self):
        proposals = []
        # each robot chooses action
        for r in self.robots:
            s = r.observe()
            a_idx = r.select_action(s)
            (tx, ty), invalid = r.propose_move(a_idx)
            proposals.append({
                'robot': r,
                's': s,
                'a': a_idx,
                'from': tuple(r.pos),
                'to': (tx, ty),
                'invalid': invalid
            })

        # Collision resolution rules
        # 1) deny moves into same target (priority by id)
        target_map = defaultdict(list)
        for p in proposals:
            if not p['invalid']:
                target_map[p['to']].append(p)

        blocked = set()
        for tgt, plist in target_map.items():
            if len(plist) > 1:
                plist_sorted = sorted(plist, key=lambda x: x['robot'].id)
                # winner = first; losers blocked
                for loser in plist_sorted[1:]:
                    blocked.add(loser['robot'].id)

        # 2) prevent swaps: A->B and B->A simultaneously
        from_to = {p['from']: p['to'] for p in proposals}
        for p in proposals:
            if p['to'] in from_to and from_to[p['to']] == p['from'] and p['from'] != p['to']:
                blocked.add(p['robot'].id)
                # block the other robot too
                other = next(q for q in proposals if q['from'] == p['to'])
                blocked.add(other['robot'].id)

        # 3) Reserve targets to avoid multiple robots moving into newly visited cells in same step:
        reserved = set()
        # apply proposals (compute rewards, but commit at end)
        next_positions = {}
        rewards = {}
        for p in proposals:
            r = p['robot']
            s = p['s']
            a_idx = p['a']
            (tx, ty) = p['to']
            if p['invalid']:
                reward = -1.0
                next_positions[r.id] = tuple(r.pos)
            elif r.id in blocked:
                reward = -2.0   # heavier penalty for collision/swap blocking
                next_positions[r.id] = tuple(r.pos)
            else:
                # if another robot already reserved target in this loop -> block (should be rare due to target_map)
                if (tx, ty) in reserved:
                    reward = -0.5
                    next_positions[r.id] = tuple(r.pos)
                else:
                    # valid move; check if new globally
                    if not self.global_visited[tx, ty]:
                        reward = 1.0
                    else:
                        reward = -0.1
                    reserved.add((tx, ty))
                    next_positions[r.id] = (tx, ty)

            rewards[r.id] = reward

        # commit moves: ensure no two robots final positions are identical (safety)
        final_positions = {}
        used = set()
        for r in self.robots:
            np_ = next_positions[r.id]
            if np_ in used:
                # conflict: keep robot in place and penalize
                final_positions[r.id] = tuple(r.pos)
                rewards[r.id] -= 0.5
            else:
                final_positions[r.id] = np_
                used.add(np_)

        # update robots state, local/global visited, and Q update
        for p in proposals:
            r = p['robot']
            s = p['s']
            a_idx = p['a']
            newpos = final_positions[r.id]
            moved = (newpos != tuple(r.pos))
            # update position
            r.pos = newpos
            gx, gy = int(newpos[0]), int(newpos[1])
            r.local_visited[gx, gy] = True
            # global visited updated immediately so other robots see it next step
            self.global_visited[gx, gy] = True

            # log to mongo if present
            if self.storage:
                floor = gy // self.env.W
                self.storage.log_step(r.id, (int(gx), int(gy)), ACTIONS[a_idx], float(rewards[r.id]), int(self.step_count))
                # also visited collection
                self.storage.log_visit(r.id, int(gx), int(gy), int(self.step_count), floor=floor)

            # Q update
            s_next = r.observe()
            r.q_update(s, a_idx, rewards[r.id], s_next)

        self.step_count += 1
        # optional sync
        if self.step_count % self.sync_K == 0:
            self.sync_maps()

        return rewards

    def sync_maps(self):
        # Fuse local visited maps -> global visited -> copy back to all local
        fused = np.zeros_like(self.global_visited)
        for r in self.robots:
            fused |= r.local_visited
        self.global_visited = fused.copy()
        for r in self.robots:
            r.local_visited = fused.copy()

    def coverage(self):
        free = (self.env.grid == 0)
        visited = np.sum(np.logical_and(self.global_visited, free))
        total = np.sum(free)
        return int(visited), int(total)

    # Export Q-tables for all robots (dict id->bytes)
    def snapshot_q_tables(self):
        return {r.id: r.get_q_bytes() for r in self.robots}

    # load q tables from dict id->bytes
    def load_q_tables(self, d):
        for r in self.robots:
            if r.id in d:
                r.load_q_bytes(d[r.id])
