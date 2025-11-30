# src/storage/mongo_client.py

from pymongo import MongoClient, ASCENDING
import pickle
import os
import numpy as np

class MongoStorage:
    def __init__(self, uri=None, db_name=None):
        uri = uri or os.getenv("MONGO_URI", "mongodb://mongo:27017/")
        db_name = db_name or os.getenv("MONGO_DB", "robotdb")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        # Ensure indexes for faster queries
        try:
            self.db.visited_cells.create_index([("gx", ASCENDING), ("gy", ASCENDING)])
            self.db.visited_cells.create_index([("robot_id", ASCENDING)])
            self.db.robots.create_index([("robot_id", ASCENDING)])
            self.db.episodes.create_index([("episode", ASCENDING)])
            self.db.q_tables.create_index([("episode", ASCENDING)])
        except Exception:
            pass
        print(f"[MongoDB] Connected to {uri}, db={db_name}")

    # ---------- Logging ----------
    def log_visit(self, robot_id, gx, gy, step, floor=None):
        doc = {"robot_id": int(robot_id), "gx": int(gx), "gy": int(gy), "step": int(step)}
        if floor is not None:
            doc["floor"] = int(floor)
        self.db.visited_cells.insert_one(doc)

    def log_step(self, robot_id, pos, action, reward, step):
        self.db.robots.insert_one({
            "robot_id": int(robot_id),
            "pos": (int(pos[0]), int(pos[1])),
            "action": str(action),
            "reward": float(reward),
            "step": int(step)
        })

    def log_episode(self, ep_number, total_reward, explored_cells, extra=None):
        doc = {"episode": int(ep_number), "total_reward": float(total_reward), "explored_cells": int(explored_cells)}
        if extra is not None:
            doc.update(extra)
        self.db.episodes.insert_one(doc)

    # ---------- Q-table storage ----------
    def save_q_table(self, episode, robot_id, q_bytes):
        self.db.q_tables.insert_one({
            "episode": int(episode),
            "robot_id": int(robot_id),
            "q_bytes": q_bytes
        })

    def load_latest_q_tables(self, episode=None):
        """
        Return dict robot_id -> latest q_bytes for a given episode.
        If episode is None, returns latest per robot.
        """
        query = {}
        if episode is not None:
            query["episode"] = int(episode)
        # aggregate to get latest per robot
        pipeline = [
            {"$match": query},
            {"$sort": {"episode": -1, "_id": -1}},
            {"$group": {"_id": "$robot_id", "q_bytes": {"$first": "$q_bytes"}, "episode": {"$first": "$episode"}}}
        ]
        out = self.db.q_tables.aggregate(pipeline)
        res = {}
        for d in out:
            res[int(d["_id"])] = d["q_bytes"]
        return res

    # ---------- Queries & utilities ----------
    def clear_collections(self, confirm=False):
        if not confirm:
            raise RuntimeError("confirm=True required to clear DB")
        self.db.visited_cells.delete_many({})
        self.db.robots.delete_many({})
        self.db.episodes.delete_many({})
        self.db.q_tables.delete_many({})

    def visited_count_map(self, H, unfolded_W):
        """
        returns a numpy array shape (H, unfolded_W) with visit counts
        """
        pipeline = [
            {"$group": {"_id": {"gx": "$gx", "gy": "$gy"}, "count": {"$sum": 1}}}
        ]
        counts = self.db.visited_cells.aggregate(pipeline)
        arr = np.zeros((H, unfolded_W), dtype=int)
        for d in counts:
            gx = int(d["_id"]["gx"])
            gy = int(d["_id"]["gy"])
            if 0 <= gx < H and 0 <= gy < unfolded_W:
                arr[gx, gy] = int(d["count"])
        return arr

    def get_robot_trajectory(self, robot_id, episode=None, sort_by="step"):
        """
        Return list of (step, gx, gy) sorted by step.
        If episode param is present, tries to filter by step range based on episode doc (optional).
        """
        q = {"robot_id": int(robot_id)}
        cursor = self.db.visited_cells.find(q).sort("step", ASCENDING)
        traj = []
        for doc in cursor:
            traj.append((int(doc.get("step", -1)), int(doc["gx"]), int(doc["gy"])))
        return traj

    def list_episodes(self, limit=50):
        return list(self.db.episodes.find({}).sort("episode", -1).limit(limit))

    def get_episode_summary(self, episode):
        return self.db.episodes.find_one({"episode": int(episode)})

    # small helper to get raw visited docs (for plotting)
    def get_visited_docs(self, episode=None, limit=100000):
        q = {}
        return list(self.db.visited_cells.find(q).limit(limit))
