#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessing and reward shaping for the DIY agent.
DIY 智能体的特征处理与奖励塑形。
"""

import math
import numpy as np

from agent_diy.conf.conf import Config


DIRECTION_TO_VEC = {
    0: (0, 0),
    1: (1, 0),
    2: (1, -1),
    3: (0, -1),
    4: (-1, -1),
    5: (-1, 0),
    6: (-1, 1),
    7: (0, 1),
    8: (1, 1),
}

ACTION_TO_DELTA = {
    0: (1, 0),
    1: (1, -1),
    2: (0, -1),
    3: (-1, -1),
    4: (-1, 0),
    5: (-1, 1),
    6: (0, 1),
    7: (1, 1),
    8: (1, 0),
    9: (1, -1),
    10: (0, -1),
    11: (-1, -1),
    12: (-1, 0),
    13: (-1, 1),
    14: (0, 1),
    15: (1, 1),
}

DIST_BUCKET_CENTER = {
    0: 15.0,
    1: 45.0,
    2: 75.0,
    3: 105.0,
    4: 135.0,
    5: 165.0,
}


def _clip_norm(value, upper, lower=0.0):
    value = float(np.clip(value, lower, upper))
    if upper - lower < 1e-6:
        return 0.0
    return (value - lower) / (upper - lower)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _get_hero(frame_state):
    hero = frame_state.get("heroes", {})
    if isinstance(hero, list):
        return hero[0] if hero else {}
    return hero if isinstance(hero, dict) else {}


def _position(entity):
    pos = entity.get("pos", {})
    return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))


def _direction_to_unit(direction):
    dx, dz = DIRECTION_TO_VEC.get(int(direction), (0, 0))
    norm = math.sqrt(dx * dx + dz * dz)
    if norm < 1e-6:
        return 0.0, 0.0
    return dx / norm, dz / norm


def _relative_from_entity(entity, hero_x, hero_z):
    visible = float(entity.get("is_in_view", 0))
    if "pos" in entity:
        ex, ez = _position(entity)
        dx = ex - hero_x
        dz = ez - hero_z
        dist = math.sqrt(dx * dx + dz * dz)
        visible = 1.0
    else:
        direction = int(entity.get("hero_relative_direction", 0))
        dx_u, dz_u = _direction_to_unit(direction)
        dist = DIST_BUCKET_CENTER.get(int(entity.get("hero_l2_distance", 5)), Config.MAX_DISTANCE)
        dx = dx_u * dist
        dz = dz_u * dist

    dir_x, dir_z = _direction_to_unit(entity.get("hero_relative_direction", 0))
    if abs(dir_x) < 1e-6 and abs(dir_z) < 1e-6 and dist > 1e-6:
        dir_x = dx / dist
        dir_z = dz / dist

    return {
        "visible": visible,
        "dx": dx,
        "dz": dz,
        "dist": dist,
        "dir_x": dir_x,
        "dir_z": dir_z,
        "speed": float(entity.get("speed", 0.0)),
    }


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_total_score = None
        self.prev_potential = 0.0

    def feature_process(self, env_obs, last_action=-1):
        observation = env_obs["observation"]
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info", [])
        legal_act_raw = observation.get("legal_action", observation.get("legal_act", []))

        hero = _get_hero(frame_state)
        hero_x, hero_z = _position(hero)

        occupancy = self._build_occupancy(map_info)
        monsters = [_relative_from_entity(m, hero_x, hero_z) for m in _as_list(frame_state.get("monsters", []))[:2]]
        while len(monsters) < 2:
            monsters.append(
                {
                    "visible": 0.0,
                    "dx": 0.0,
                    "dz": 0.0,
                    "dist": Config.MAX_DISTANCE,
                    "dir_x": 0.0,
                    "dir_z": 0.0,
                    "speed": 0.0,
                }
            )

        organs = [o for o in _as_list(frame_state.get("organs", [])) if int(o.get("status", 1)) == 1]
        treasure_entities = [o for o in organs if int(o.get("sub_type", 0)) == 1]
        buff_entities = [o for o in organs if int(o.get("sub_type", 0)) == 2]

        treasure_channel = self._build_object_channel(treasure_entities, hero_x, hero_z)
        buff_channel = self._build_object_channel(buff_entities, hero_x, hero_z)
        monster_threat = self._build_monster_threat_channel(monsters, hero_x, hero_z)

        nearest_treasure = self._nearest_entity_feature(treasure_entities, hero_x, hero_z)
        nearest_buff = self._nearest_entity_feature(buff_entities, hero_x, hero_z)

        legal_action = self._parse_legal_action(legal_act_raw, hero)
        action_priors = self._build_action_priors(
            occupancy=occupancy,
            treasure_channel=treasure_channel,
            buff_channel=buff_channel,
            monster_threat=monster_threat,
            monsters=monsters,
            hero=hero,
            legal_action=legal_action,
        )

        vector_feature, summary = self._build_vector_feature(
            hero=hero,
            env_info=env_info,
            monsters=monsters,
            nearest_treasure=nearest_treasure,
            nearest_buff=nearest_buff,
            occupancy=occupancy,
        )

        map_feature = np.stack(
            [
                occupancy,
                treasure_channel,
                buff_channel,
                monster_threat,
            ],
            axis=0,
        ).astype(np.float32)

        feature = np.concatenate(
            [
                vector_feature,
                map_feature.reshape(-1),
                action_priors,
            ]
        ).astype(np.float32)

        current_total_score = float(env_info.get("total_score", 0.0))
        potential = self._compute_potential(summary)
        if self.prev_total_score is None:
            score_reward = 0.0
            shaping_reward = 0.0
        else:
            score_reward = np.clip(
                (current_total_score - self.prev_total_score) / Config.SCORE_REWARD_SCALE,
                -4.0,
                4.0,
            )
            shaping_reward = Config.GAMMA * potential - self.prev_potential

        reward = score_reward + shaping_reward
        self.prev_total_score = current_total_score
        self.prev_potential = potential

        remain_info = {
            "reward": [reward],
            "score_reward": [score_reward],
            "shaping_reward": [shaping_reward],
            "potential": [potential],
        }
        return feature, legal_action, remain_info

    def _build_occupancy(self, map_info):
        map_array = np.asarray(map_info, dtype=np.float32)
        if map_array.ndim != 2:
            return np.ones((Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32)

        rows, cols = map_array.shape
        if rows == Config.LOCAL_MAP_SIZE and cols == Config.LOCAL_MAP_SIZE:
            return (map_array != 0).astype(np.float32)

        canvas = np.zeros((Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32)
        min_r = min(rows, Config.LOCAL_MAP_SIZE)
        min_c = min(cols, Config.LOCAL_MAP_SIZE)
        r0 = max((Config.LOCAL_MAP_SIZE - min_r) // 2, 0)
        c0 = max((Config.LOCAL_MAP_SIZE - min_c) // 2, 0)
        canvas[r0 : r0 + min_r, c0 : c0 + min_c] = (map_array[:min_r, :min_c] != 0).astype(np.float32)
        return canvas

    def _build_object_channel(self, entities, hero_x, hero_z):
        channel = np.zeros((Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32)
        center = Config.LOCAL_MAP_RADIUS
        for entity in entities:
            if "pos" not in entity:
                continue
            ex, ez = _position(entity)
            dx = int(round(ex - hero_x))
            dz = int(round(ez - hero_z))
            if abs(dx) > Config.LOCAL_MAP_RADIUS or abs(dz) > Config.LOCAL_MAP_RADIUS:
                continue
            row = center + dz
            col = center + dx
            if 0 <= row < Config.LOCAL_MAP_SIZE and 0 <= col < Config.LOCAL_MAP_SIZE:
                channel[row, col] = 1.0
        return channel

    def _build_monster_threat_channel(self, monsters, hero_x, hero_z):
        center = Config.LOCAL_MAP_RADIUS
        channel = np.zeros((Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE), dtype=np.float32)
        for monster in monsters:
            if monster["visible"] < 0.5:
                continue
            mx = int(round(monster["dx"]))
            mz = int(round(monster["dz"]))
            row = center + mz
            col = center + mx
            if row < 0 or row >= Config.LOCAL_MAP_SIZE or col < 0 or col >= Config.LOCAL_MAP_SIZE:
                continue
            sigma = 2.0 + min(monster["speed"], Config.MAX_MONSTER_SPEED) * 0.6
            for r in range(Config.LOCAL_MAP_SIZE):
                for c in range(Config.LOCAL_MAP_SIZE):
                    dist = math.sqrt((r - row) ** 2 + (c - col) ** 2)
                    channel[r, c] = max(channel[r, c], math.exp(-dist / max(sigma, 1.0)))
        return channel

    def _nearest_entity_feature(self, entities, hero_x, hero_z):
        best = {
            "visible": 0.0,
            "dx": 0.0,
            "dz": 0.0,
            "dist": Config.MAX_DISTANCE,
            "dir_x": 0.0,
            "dir_z": 0.0,
        }
        for entity in entities:
            rel = _relative_from_entity(entity, hero_x, hero_z)
            if rel["dist"] < best["dist"]:
                best = rel
        return best

    def _parse_legal_action(self, legal_act_raw, hero):
        flash_ready = float(hero.get("flash_cooldown", 0.0)) <= 1e-6
        legal_action = [1] * 8 + [int(flash_ready)] * 8

        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for idx in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                    legal_action[idx] = int(legal_act_raw[idx])
            else:
                valid = {int(item) for item in legal_act_raw if 0 <= int(item) < Config.ACTION_NUM}
                legal_action = [1 if idx in valid else 0 for idx in range(Config.ACTION_NUM)]

        if sum(legal_action[:8]) == 0:
            legal_action[:8] = [1] * 8
        return legal_action

    def _build_vector_feature(self, hero, env_info, monsters, nearest_treasure, nearest_buff, occupancy):
        hero_x, hero_z = _position(hero)
        step_no = float(env_info.get("step_no", env_info.get("finished_steps", 0.0)))
        max_step = float(env_info.get("max_step", env_info.get("max_steps", 1000.0)))

        total_score = float(env_info.get("total_score", 0.0))
        step_score = float(env_info.get("step_score", 0.0))
        treasure_score = float(env_info.get("treasure_score", 0.0))

        buff_remaining = float(hero.get("buff_remaining_time", 0.0))
        flash_cd = float(hero.get("flash_cooldown", 0.0))
        treasures_collected = float(env_info.get("treasures_collected", hero.get("treasure_collected_count", 0.0)))
        total_treasure = max(float(env_info.get("total_treasure", 10.0)), 1.0)
        collected_buff = float(env_info.get("collected_buff", 0.0))
        total_buff = max(float(env_info.get("total_buff", 2.0)), 1.0)

        step_norm = _clip_norm(step_no, max_step if max_step > 0 else 1000.0)
        remain_step_norm = 1.0 - step_norm
        hero_feature = np.array(
            [
                _clip_norm(hero_x, Config.MAP_SIZE),
                _clip_norm(hero_z, Config.MAP_SIZE),
                step_norm,
                remain_step_norm,
                _clip_norm(flash_cd, Config.MAX_FLASH_CD),
                1.0 if flash_cd <= 1e-6 else 0.0,
                _clip_norm(buff_remaining, Config.MAX_BUFF_DURATION),
                1.0 if buff_remaining > 0 else 0.0,
                _clip_norm(total_score, Config.MAX_TOTAL_SCORE),
                _clip_norm(step_score, Config.MAX_TOTAL_SCORE),
                _clip_norm(treasure_score, Config.MAX_TOTAL_SCORE),
                _clip_norm(max_step, 2000.0),
            ],
            dtype=np.float32,
        )

        monster_feature = []
        min_monster_dist_norm = 1.0
        for monster in monsters[:2]:
            dist_norm = _clip_norm(monster["dist"], Config.MAX_DISTANCE)
            min_monster_dist_norm = min(min_monster_dist_norm, dist_norm)
            monster_feature.extend(
                [
                    monster["visible"],
                    float(np.clip(monster["dx"] / Config.MAP_SIZE, -1.0, 1.0)),
                    float(np.clip(monster["dz"] / Config.MAP_SIZE, -1.0, 1.0)),
                    dist_norm,
                    monster["dir_x"],
                    monster["dir_z"],
                    _clip_norm(monster["speed"], Config.MAX_MONSTER_SPEED),
                    1.0 if monster["dist"] <= 12.0 else 0.0,
                ]
            )

        nearest_treasure_dist_norm = _clip_norm(nearest_treasure["dist"], Config.MAX_DISTANCE)
        treasure_feature = np.array(
            [
                nearest_treasure["visible"],
                float(np.clip(nearest_treasure["dx"] / Config.MAP_SIZE, -1.0, 1.0)),
                float(np.clip(nearest_treasure["dz"] / Config.MAP_SIZE, -1.0, 1.0)),
                nearest_treasure_dist_norm,
                nearest_treasure["dir_x"],
                nearest_treasure["dir_z"],
                _clip_norm(total_treasure - treasures_collected, total_treasure),
            ],
            dtype=np.float32,
        )

        buff_needed = 1.0 if buff_remaining <= 1e-6 and min_monster_dist_norm < 0.35 else 0.0
        nearest_buff_dist_norm = _clip_norm(nearest_buff["dist"], Config.MAX_DISTANCE)
        buff_feature = np.array(
            [
                nearest_buff["visible"],
                float(np.clip(nearest_buff["dx"] / Config.MAP_SIZE, -1.0, 1.0)),
                float(np.clip(nearest_buff["dz"] / Config.MAP_SIZE, -1.0, 1.0)),
                nearest_buff_dist_norm,
                nearest_buff["dir_x"],
                nearest_buff["dir_z"],
                buff_needed,
            ],
            dtype=np.float32,
        )

        center = Config.LOCAL_MAP_RADIUS
        open_area = occupancy[max(center - 2, 0) : center + 3, max(center - 2, 0) : center + 3].mean()
        open_neighbors = 0
        for action in range(8):
            dr, dc = ACTION_TO_DELTA[action][1], ACTION_TO_DELTA[action][0]
            nr, nc = center + dr, center + dc
            if 0 <= nr < Config.LOCAL_MAP_SIZE and 0 <= nc < Config.LOCAL_MAP_SIZE and occupancy[nr, nc] > 0.5:
                open_neighbors += 1

        env_feature = np.array(
            [
                _clip_norm(total_treasure, 10.0),
                _clip_norm(treasures_collected, total_treasure),
                _clip_norm(total_buff, 2.0),
                _clip_norm(collected_buff, total_buff),
                _clip_norm(float(env_info.get("monster_interval", 300.0)), 2000.0),
                _clip_norm(float(env_info.get("monster_speed", 1.0)), Config.MAX_MONSTER_SPEED),
                _clip_norm(float(env_info.get("flash_count", 0.0)), 100.0),
                float(open_area),
                1.0 if open_neighbors <= 2 else 0.0,
                min_monster_dist_norm,
            ],
            dtype=np.float32,
        )

        ray_feature = np.array(self._ray_distances(occupancy), dtype=np.float32)

        vector_feature = np.concatenate(
            [
                hero_feature,
                np.asarray(monster_feature, dtype=np.float32),
                treasure_feature,
                buff_feature,
                env_feature,
                ray_feature,
            ]
        ).astype(np.float32)

        summary = {
            "safe": min_monster_dist_norm,
            "treasure": nearest_treasure["visible"] * (1.0 - nearest_treasure_dist_norm),
            "buff": buff_needed * nearest_buff["visible"] * (1.0 - nearest_buff_dist_norm),
            "mobility": float(open_area),
        }
        return vector_feature, summary

    def _ray_distances(self, occupancy):
        center = Config.LOCAL_MAP_RADIUS
        ray_values = []
        for action in range(8):
            dx, dz = ACTION_TO_DELTA[action]
            row, col = center, center
            steps = 0
            while True:
                nr = row + dz
                nc = col + dx
                if nr < 0 or nr >= Config.LOCAL_MAP_SIZE or nc < 0 or nc >= Config.LOCAL_MAP_SIZE:
                    break
                if occupancy[nr, nc] <= 0.5:
                    break
                row, col = nr, nc
                steps += 1
            ray_values.append(steps / float(Config.LOCAL_MAP_RADIUS))
        return ray_values

    def _compute_potential(self, summary):
        potential = (
            0.50 * summary["safe"]
            + 0.20 * summary["treasure"]
            + 0.10 * summary["buff"]
            + 0.20 * summary["mobility"]
        )
        return float(np.clip(potential, 0.0, 1.0))

    def _build_action_priors(
        self,
        occupancy,
        treasure_channel,
        buff_channel,
        monster_threat,
        monsters,
        hero,
        legal_action,
    ):
        center = Config.LOCAL_MAP_RADIUS
        move_speed = 2 if float(hero.get("buff_remaining_time", 0.0)) > 0 else 1
        priors = np.full((Config.ACTION_NUM,), -1.0, dtype=np.float32)

        monster_positions = []
        for monster in monsters:
            if monster["visible"] < 0.5:
                continue
            row = center + int(round(monster["dz"]))
            col = center + int(round(monster["dx"]))
            if 0 <= row < Config.LOCAL_MAP_SIZE and 0 <= col < Config.LOCAL_MAP_SIZE:
                monster_positions.append((row, col))

        default_clearance = 1.0 - min(_clip_norm(m["dist"], Config.MAX_DISTANCE) for m in monsters)
        default_clearance = 1.0 - float(np.clip(default_clearance, 0.0, 1.0))

        for action in range(Config.ACTION_NUM):
            if legal_action[action] <= 0:
                continue

            end_pos, path = self._simulate_action(occupancy, action, move_speed)
            end_row, end_col = end_pos

            path_treasure = max([treasure_channel[r, c] for r, c in path], default=0.0)
            path_buff = max([buff_channel[r, c] for r, c in path], default=0.0)

            neighborhood = occupancy[
                max(end_row - 1, 0) : min(end_row + 2, Config.LOCAL_MAP_SIZE),
                max(end_col - 1, 0) : min(end_col + 2, Config.LOCAL_MAP_SIZE),
            ]
            end_open = float(neighborhood.mean()) if neighborhood.size else 0.0

            free_neighbors = 0
            for base_action in range(8):
                dx, dz = ACTION_TO_DELTA[base_action]
                nr = end_row + dz
                nc = end_col + dx
                if 0 <= nr < Config.LOCAL_MAP_SIZE and 0 <= nc < Config.LOCAL_MAP_SIZE and occupancy[nr, nc] > 0.5:
                    free_neighbors += 1

            if monster_positions:
                min_dist = min(math.sqrt((end_row - mr) ** 2 + (end_col - mc) ** 2) for mr, mc in monster_positions)
                clearance = float(np.clip(min_dist / 12.0, 0.0, 1.0))
            else:
                clearance = default_clearance

            threat = float(monster_threat[end_row, end_col])
            deadend = 1.0 if free_neighbors <= 2 else 0.0
            stay_penalty = 1.0 if (end_row == center and end_col == center) else 0.0
            flash_cost = 0.1 if action >= 8 else 0.0

            utility = (
                1.8 * clearance
                + 1.2 * path_treasure
                + 0.8 * path_buff
                + 0.6 * end_open
                - 1.5 * threat
                - 1.0 * deadend
                - 1.2 * stay_penalty
                - flash_cost
            )
            priors[action] = math.tanh(utility)

        return priors.astype(np.float32)

    def _simulate_action(self, occupancy, action, move_speed):
        center = Config.LOCAL_MAP_RADIUS
        row, col = center, center
        dx, dz = ACTION_TO_DELTA[action]

        if action < 8:
            max_steps = move_speed
            path = []
            for _ in range(max_steps):
                nr = row + dz
                nc = col + dx
                if not self._can_step(occupancy, row, col, nr, nc):
                    break
                row, col = nr, nc
                path.append((row, col))
            return (row, col), path

        max_steps = 10 if action in (8, 10, 12, 14) else 8
        path = []
        for _ in range(max_steps):
            nr = row + dz
            nc = col + dx
            if not self._can_step(occupancy, row, col, nr, nc):
                break
            row, col = nr, nc
            path.append((row, col))
        return (row, col), path

    def _can_step(self, occupancy, row, col, nr, nc):
        if nr < 0 or nr >= Config.LOCAL_MAP_SIZE or nc < 0 or nc >= Config.LOCAL_MAP_SIZE:
            return False
        if occupancy[nr, nc] <= 0.5:
            return False

        dr = nr - row
        dc = nc - col
        if abs(dr) == 1 and abs(dc) == 1:
            side_a = (row, nc)
            side_b = (nr, col)
            side_a_ok = 0 <= side_a[0] < Config.LOCAL_MAP_SIZE and 0 <= side_a[1] < Config.LOCAL_MAP_SIZE
            side_b_ok = 0 <= side_b[0] < Config.LOCAL_MAP_SIZE and 0 <= side_b[1] < Config.LOCAL_MAP_SIZE
            side_a_pass = side_a_ok and occupancy[side_a[0], side_a[1]] > 0.5
            side_b_pass = side_b_ok and occupancy[side_b[0], side_b[1]] > 0.5
            if not (side_a_pass or side_b_pass):
                return False
        return True
