# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PUCT sampler and Erdős State mirror ttt-discover-ref:
#   ttt_discover/tinker_utils/sampler.py (PUCTSampler)
#   ttt_discover/tinker_utils/state.py (State)
#   examples/erdos_min_overlap/env.py (value = -C₅, minimize)

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


def to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    return obj


@contextmanager
def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()}\n{time.time()}\n".encode("utf-8"))
            finally:
                os.close(fd)
            break
        except FileExistsError:
            try:
                st = os.stat(lock_path)
                if (time.time() - st.st_mtime) > stale_s:
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(poll_s)
    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(obj), f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_json_or_default(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def _sampler_file_for_step(base_path: str, step: int) -> str:
    base_name = base_path.replace(".json", "")
    return f"{base_name}_step_{step:06d}.json"


@dataclass
class ErdosRefState:
    """Mirrors ttt_discover State for Erdős (value = -C₅, higher is better)."""

    timestep: int
    construction: list
    code: str = ""
    value: Optional[float] = None
    parent_values: list[float] = field(default_factory=list)
    parents: list[dict] = field(default_factory=list)
    observation: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "type": "ErdosRefState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": list(self.parent_values),
            "parents": list(self.parents),
            "observation": self.observation,
            "construction": to_json_serializable(self.construction),
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ErdosRefState:
        return cls(
            timestep=int(d["timestep"]),
            construction=list(d.get("construction") or []),
            code=str(d.get("code") or ""),
            value=d.get("value"),
            parent_values=list(d.get("parent_values") or []),
            parents=list(d.get("parents") or []),
            observation=str(d.get("observation") or ""),
            id=str(d.get("id") or uuid.uuid4()),
        )


def erdos_ref_state_to_prompt_state(s: ErdosRefState) -> dict[str, Any]:
    """Map to keys consumed by build_erdos_question / state_to_prompt."""
    c5 = -float(s.value) if s.value is not None else None
    parent_c5 = None
    if s.parent_values:
        parent_c5 = -float(s.parent_values[0])
    return {
        "construction": list(s.construction) if s.construction else [],
        "c5_bound": c5,
        "n_points": len(s.construction) if s.construction else 0,
        "code": s.code or "",
        "parent_c5": parent_c5,
        "observation": s.observation or "",
    }


class ErdosRefPUCTSampler:
    """
    Line-for-line behavior of ttt_discover PUCTSampler for Erdős discovery.

    score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n[i])
    Q(i) = m[i] if n[i]>0 else R(i); P = rank prior; scale = max(R)-min(R) on non-initial.
    """

    def __init__(
        self,
        file_path: str,
        init_state_fn: Callable[[], ErdosRefState],
        max_buffer_size: int = 1000,
        batch_size: int = 1,
        resume_step: Optional[int] = None,
        puct_c: float = 1.0,
        topk_children: int = 2,
        max_construction_len: Optional[int] = 1000,
    ):
        self.file_path = file_path
        self._init_state_fn = init_state_fn
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.topk_children = topk_children
        self.puct_c = float(puct_c)
        self.max_construction_len = max_construction_len

        self._states: list[ErdosRefState] = []
        self._initial_states: list[ErdosRefState] = []
        self._last_sampled_states: list[ErdosRefState] = []
        self._last_sampled_indices: list[int] = []
        self._lock = threading.Lock()
        self._current_step = resume_step if resume_step is not None else 0

        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T: int = 0
        self._last_scale: float = 1.0
        self._last_puct_stats: list[tuple[int, float, float, float, float]] = []

        if resume_step is not None:
            self._load(resume_step)
        if not self._states:
            for _ in range(batch_size):
                state = init_state_fn()
                self._initial_states.append(state)
                self._states.append(state)
            self._save(self._current_step)

    @staticmethod
    def _set_parent_info(child: ErdosRefState, parent: ErdosRefState) -> None:
        child.parent_values = (
            [parent.value] + parent.parent_values if parent.value is not None else []
        )
        child.parents = [{"id": parent.id, "timestep": parent.timestep}] + parent.parents

    @staticmethod
    def _filter_topk_per_parent(
        states: list[ErdosRefState],
        parent_states: list[ErdosRefState],
        k: int,
    ) -> tuple[list[ErdosRefState], list[ErdosRefState]]:
        if not states:
            return [], []
        if k == 0:
            return states, parent_states
        parent_to_children: dict[str, list[tuple[ErdosRefState, ErdosRefState]]] = {}
        for child, parent in zip(states, parent_states):
            pid = parent.id
            parent_to_children.setdefault(pid, []).append((child, parent))
        topk_children, topk_parents = [], []
        for children_and_parents in parent_to_children.values():
            sorted_pairs = sorted(
                children_and_parents,
                key=lambda x: x[0].value if x[0].value is not None else float("-inf"),
                reverse=True,
            )
            for child, parent in sorted_pairs[:k]:
                topk_children.append(child)
                topk_parents.append(parent)
        return topk_children, topk_parents

    def _load(self, step: int) -> None:
        file_path = _sampler_file_for_step(self.file_path, step)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Cannot resume from step {step}: sampler file not found: {file_path}"
            )
        with _file_lock(f"{file_path}.lock"):
            store = _read_json_or_default(file_path, default=None)
        if store is None:
            raise ValueError(f"Failed to load sampler state from {file_path}")
        self._states = [ErdosRefState.from_dict(s) for s in store.get("states", [])]
        self._initial_states = [
            ErdosRefState.from_dict(s) for s in store.get("initial_states", [])
        ]
        self._n = store.get("puct_n", {}) or {}
        self._m = store.get("puct_m", {}) or {}
        self._T = int(store.get("puct_T", 0) or 0)

    def _save(self, step: int) -> None:
        save_path = _sampler_file_for_step(self.file_path, step)
        store = {
            "step": step,
            "states": [s.to_dict() for s in self._states],
            "initial_states": [s.to_dict() for s in self._initial_states],
            "puct_n": self._n,
            "puct_m": self._m,
            "puct_T": self._T,
        }
        with _file_lock(f"{save_path}.lock"):
            _atomic_write_json(save_path, store)

    def _get_construction_key(self, state: ErdosRefState):
        if state.construction:
            return tuple(state.construction)
        if state.code:
            return state.code
        return None

    def _compute_scale(
        self, values: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> float:
        if values.size == 0:
            return 1.0
        v = values[mask] if mask is not None else values
        return float(max(np.max(v) - np.min(v), 1e-6)) if v.size > 0 else 1.0

    def _compute_prior(self, values: np.ndarray, scale: float) -> np.ndarray:
        del scale  # matches ref signature
        if values.size == 0:
            return np.array([])
        n = len(values)
        ranks = np.argsort(np.argsort(-values))
        weights = (n - ranks).astype(np.float64)
        return weights / weights.sum()

    def _get_lineage(self, state: ErdosRefState) -> set[str]:
        lineage = {state.id}
        for p in state.parents or []:
            if p.get("id"):
                lineage.add(str(p["id"]))
        return lineage

    def _build_children_map(self) -> dict[str, set[str]]:
        children: dict[str, set[str]] = {}
        for s in self._states:
            for p in s.parents or []:
                pid = p.get("id")
                if pid:
                    children.setdefault(str(pid), set()).add(s.id)
        return children

    def _get_full_lineage(
        self, state: ErdosRefState, children_map: dict[str, set[str]]
    ) -> set[str]:
        lineage = self._get_lineage(state)
        queue = [state.id]
        visited = {state.id}
        while queue:
            sid = queue.pop(0)
            for child_id in children_map.get(sid, []):
                if child_id not in visited:
                    visited.add(child_id)
                    lineage.add(child_id)
                    queue.append(child_id)
        return lineage

    def sample_states(self, num_states: int) -> list[ErdosRefState]:
        initial_ids = {s.id for s in self._initial_states}
        candidates = list(self._states)

        if not candidates:
            picked = [self._init_state_fn() for _ in range(num_states)]
            self._last_sampled_states = picked
            self._last_sampled_indices = []
            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
            return picked

        vals = np.array(
            [float(s.value if s.value is not None else float("-inf")) for s in candidates]
        )
        non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
        scale = self._compute_scale(
            vals, non_initial_mask if non_initial_mask.any() else None
        )
        self._last_scale = scale
        p = self._compute_prior(vals, scale)
        sqrt_t = np.sqrt(1.0 + self._T)

        scores = []
        for i, s in enumerate(candidates):
            n = self._n.get(s.id, 0)
            m = self._m.get(s.id, vals[i])
            q = m if n > 0 else vals[i]
            bonus = self.puct_c * scale * p[i] * sqrt_t / (1.0 + n)
            score = q + bonus
            scores.append((score, vals[i], s, n, q, p[i], bonus))

        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if num_states > 1:
            children_map = self._build_children_map()
            picked, top_scores = [], []
            blocked_ids: set[str] = set()
            for entry in scores:
                s = entry[2]
                if s.id in blocked_ids:
                    continue
                picked.append(s)
                top_scores.append(entry)
                blocked_ids.update(self._get_full_lineage(s, children_map))
                if len(picked) >= num_states:
                    break
        else:
            top_scores = scores[:num_states]
            picked = [t[2] for t in top_scores]

        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
        self._last_sampled_states = picked
        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]

        # Erdős: no construction_length_limits — no refresh (ref no-op for this env)
        return picked

    def update_states(
        self,
        states: list[ErdosRefState],
        parent_states: list[ErdosRefState],
        save: bool = True,
        step: Optional[int] = None,
    ) -> None:
        if not states:
            return
        assert len(states) == len(parent_states)

        parent_max: dict[str, float] = {}
        parent_obj: dict[str, ErdosRefState] = {}
        for child, parent in zip(states, parent_states):
            if child.value is None:
                continue
            pid = parent.id
            parent_obj[pid] = parent
            parent_max[pid] = max(parent_max.get(pid, float("-inf")), float(child.value))

        for pid, y in parent_max.items():
            self._m[pid] = max(self._m.get(pid, y), y)
            parent = parent_obj[pid]
            anc_ids = [pid] + [
                str(p["id"]) for p in (parent.parents or []) if p.get("id")
            ]
            for aid in anc_ids:
                self._n[aid] = self._n.get(aid, 0) + 1
            self._T += 1

        states, parent_states = self._filter_topk_per_parent(
            states, parent_states, self.topk_children
        )
        existing = {self._get_construction_key(s) for s in self._states}
        existing.discard(None)

        new_states = []
        for child, parent in zip(states, parent_states):
            if child.value is None:
                continue
            if (
                self.max_construction_len is not None
                and child.construction
                and len(child.construction) > self.max_construction_len
            ):
                continue
            key = self._get_construction_key(child)
            if key is not None and key in existing:
                continue
            self._set_parent_info(child, parent)
            new_states.append(child)
            if key is not None:
                existing.add(key)

        if not new_states:
            return
        with self._lock:
            self._states.extend(new_states)
            if save:
                self._finalize_and_save(step)

    def _finalize_and_save(self, step: Optional[int] = None) -> None:
        if len(self._states) > self.max_buffer_size:
            actual_values = [
                s.value if s.value is not None else float("-inf") for s in self._states
            ]
            by_actual = list(np.argsort(actual_values)[::-1])
            initial_ids = {s.id for s in self._initial_states}
            initial_indices = {i for i, s in enumerate(self._states) if s.id in initial_ids}
            keep = set(initial_indices)
            for i in by_actual:
                if len(keep) >= self.max_buffer_size:
                    break
                keep.add(i)
            self._states = [self._states[i] for i in sorted(keep)]
        if step is not None:
            self._current_step = step
        self._save(self._current_step)

    def flush(self, step: Optional[int] = None) -> None:
        with self._lock:
            if self.topk_children > 0:
                by_parent: dict[str, list[ErdosRefState]] = {}
                no_parent: list[ErdosRefState] = []
                for s in self._states:
                    pid = s.parents[0]["id"] if s.parents else None
                    if pid:
                        by_parent.setdefault(pid, []).append(s)
                    else:
                        no_parent.append(s)
                filtered = []
                for children in by_parent.values():
                    children.sort(
                        key=lambda x: x.value if x.value is not None else float("-inf"),
                        reverse=True,
                    )
                    filtered.extend(children[: self.topk_children])
                self._states = no_parent + filtered
            self._finalize_and_save(step)

    def record_failed_rollout(self, parent: ErdosRefState) -> None:
        anc_ids = [parent.id] + [
            str(p["id"]) for p in (parent.parents or []) if p.get("id")
        ]
        for aid in anc_ids:
            self._n[aid] = self._n.get(aid, 0) + 1
        self._T += 1

    def get_sample_stats(self) -> dict[str, float]:
        def _stats(values, prefix):
            arr = np.array([v for v in values if v is not None])
            if len(arr) == 0:
                return {}
            return {
                f"{prefix}/mean": float(np.mean(arr)),
                f"{prefix}/std": float(np.std(arr)),
                f"{prefix}/min": float(np.min(arr)),
                f"{prefix}/max": float(np.max(arr)),
            }

        buffer_values = [s.value for s in self._states]
        buffer_timesteps = [s.timestep for s in self._states]
        buffer_constr_lens = [
            len(s.construction) if s.construction else 0 for s in self._states
        ]
        sampled_values = [s.value for s in self._last_sampled_states]
        sampled_timesteps = [s.timestep for s in self._last_sampled_states]
        sampled_constr_lens = [
            len(s.construction) if s.construction else 0
            for s in self._last_sampled_states
        ]
        stats: dict[str, float] = {
            "puct/buffer_size": float(len(self._states)),
            "puct/sampled_size": float(len(self._last_sampled_states)),
            "puct/T": float(self._T),
            "puct/scale_last": float(self._last_scale),
        }
        stats.update(_stats(buffer_values, "puct/buffer_value"))
        stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
        stats.update(_stats(buffer_constr_lens, "puct/buffer_construction_len"))
        stats.update(_stats(sampled_values, "puct/sampled_value"))
        stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
        stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
        return stats
