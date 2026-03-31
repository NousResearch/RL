"""
PUCT buffer for TTT-Discover state reuse.

Reference: "Learning to Discover at Test Time" (arXiv:2601.04116)

The buffer maintains a tree of (state, reward) nodes. At each training step,
PUCT scoring selects which states to warm-start rollouts from, balancing:
  - Exploitation: states whose children have achieved high rewards (high Q)
  - Exploration:  states that haven't been visited much yet (low n)

Pure data structure — no ML framework dependencies.
"""

import math
import dataclasses
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal node
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Node:
    state: Any
    reward: float           # reward of THIS state (from its own evaluation)
    parent_key: Any         # key of parent node, or None for roots
    children_keys: list     # keys of direct children
    n: int                  # visit count (number of times selected for expansion)
    Q: float                # max reward among all descendants (or own reward if leaf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_key(state: Any) -> Any:
    """Convert state to a hashable key.

    Supports: str, int, float, tuple, list, np.ndarray, and arbitrary objects
    (fallback: id-based, so two different objects with equal content are
    treated as distinct — acceptable for LLM response strings).
    """
    if isinstance(state, (str, int, float, bool)):
        return state
    if isinstance(state, np.ndarray):
        return (state.dtype, state.shape, state.tobytes())
    if isinstance(state, (list, tuple)):
        return tuple(_make_key(x) for x in state)
    # Fallback: identity-based key — wrap id so it doesn't collide with ints
    return ("__id__", id(state))


# ---------------------------------------------------------------------------
# PUCTBuffer
# ---------------------------------------------------------------------------

class PUCTBuffer:
    """
    Tree-structured buffer with PUCT selection.

    PUCT score for node s:
        score(s) = Q(s) + c · P(s) · sqrt(1 + T) / (1 + n(s))

    Where:
        Q(s)  = max reward among all descendants of s (own reward if leaf)
        P(s)  = rank-based prior: rank states by reward, normalize by total rank
        n(s)  = visit count of s
        T     = total visit count across all nodes
        c     = exploration constant (default 1.0)
    """

    def __init__(self, c: float = 1.0) -> None:
        self.c = c
        self._nodes: dict[Any, _Node] = {}   # key → _Node
        self._T: int = 0                      # total expansions so far

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, state: Any, reward: float, parent_state: Any = None) -> None:
        """Insert a new node into the buffer.

        If the state is already present, this is a no-op (deduplication).
        If parent_state is given and present in the buffer, the new node is
        linked as a child and Q values are propagated upward.

        Args:
            state:        The state to insert (any type with a consistent identity).
            reward:       Scalar reward associated with this state.
            parent_state: Parent state, or None for a root node.
        """
        key = _make_key(state)
        if key in self._nodes:
            return  # already present — deduplicate

        parent_key = _make_key(parent_state) if parent_state is not None else None
        node = _Node(
            state=state,
            reward=float(reward),
            parent_key=parent_key,
            children_keys=[],
            n=0,
            Q=float(reward),  # leaf: Q = own reward
        )
        self._nodes[key] = node

        if parent_key is not None and parent_key in self._nodes:
            self._nodes[parent_key].children_keys.append(key)
            self._propagate_Q(parent_key)

    def select(
        self, batch_size: int, num_groups: int = 8
    ) -> list[tuple[Any, list]]:
        """Select states to warm-start rollouts from.

        Scores each node with PUCT, picks the top `num_groups` distinct states,
        and returns `batch_size` (state, context) pairs grouped so that each
        group of `batch_size // num_groups` entries shares the same state.

        Context is the ancestry path from root to the selected node:
            [(ancestor_state, ancestor_reward), ..., (selected_state, selected_reward)]
        The env uses this to build the prompt (previous attempts / warm start).

        Visit counts are incremented for the selected nodes, and T is updated.

        Args:
            batch_size:  Total number of (state, context) pairs to return.
                         Must be divisible by num_groups.
            num_groups:  Number of distinct initial states to select.

        Returns:
            List of (state, context) tuples, length == batch_size.
        """
        if not self._nodes:
            raise ValueError("Buffer is empty — call add() before select()")
        if batch_size % num_groups != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_groups ({num_groups})"
            )
        rollouts_per_group = batch_size // num_groups

        priors = self._rank_priors()
        scores = {
            key: self._puct_score(node, priors[key])
            for key, node in self._nodes.items()
        }

        # Top num_groups keys by PUCT score (at most len(nodes) if buffer is small)
        k = min(num_groups, len(self._nodes))
        top_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]

        result: list[tuple[Any, list]] = []
        for key in top_keys:
            node = self._nodes[key]
            context = self._ancestry(key)
            pair = (node.state, context)
            result.extend([pair] * rollouts_per_group)
            # Increment visit count for this selection
            node.n += 1
            self._T += 1

        return result

    def update(
        self, parent_state: Any, child_state: Any, reward: float
    ) -> None:
        """Add a child node and update Q values up the tree.

        Convenience wrapper around add() that makes the parent/child
        relationship explicit.

        Args:
            parent_state: The state that was selected and rolled out from.
            child_state:  The resulting new state produced by the rollout.
            reward:       Reward of the new child state.
        """
        self.add(child_state, reward, parent_state=parent_state)

    def best(self) -> tuple[Any, float]:
        """Return the (state, reward) with the highest reward ever seen.

        Returns:
            (state, reward) tuple.
        """
        if not self._nodes:
            raise ValueError("Buffer is empty")
        best_key = max(self._nodes, key=lambda k: self._nodes[k].reward)
        node = self._nodes[best_key]
        return node.state, node.reward

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _puct_score(self, node: _Node, prior: float) -> float:
        return node.Q + self.c * prior * math.sqrt(1 + self._T) / (1 + node.n)

    def _rank_priors(self) -> dict[Any, float]:
        """Rank-based prior: rank by node reward, normalize by sum of ranks.

        Rank 1 = lowest reward, rank N = highest.  Ties get the same rank
        (average of tied ranks), consistent with scipy.stats.rankdata.
        """
        keys = list(self._nodes.keys())
        rewards = np.array([self._nodes[k].reward for k in keys], dtype=float)

        # argsort twice gives rank (0-indexed); add 1 to make 1-indexed
        order = np.argsort(rewards)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(rewards) + 1, dtype=float)

        # Handle ties: assign average rank to tied rewards.
        # Use ranks[tied].mean() — not tied.mean()+1, which would use array
        # indices instead of the already-assigned rank values.
        # (simple O(N²) loop is fine for buffer sizes we care about)
        for i, r in enumerate(rewards):
            tied = np.where(rewards == r)[0]
            if len(tied) > 1:
                ranks[tied] = ranks[tied].mean()

        total = ranks.sum()
        return {k: float(ranks[i] / total) for i, k in enumerate(keys)}

    def _propagate_Q(self, key: Any) -> None:
        """Propagate max-Q upward from `key` to the root."""
        node = self._nodes[key]
        if node.children_keys:
            child_rewards = [
                self._nodes[ck].Q
                for ck in node.children_keys
                if ck in self._nodes
            ]
            new_Q = max(node.reward, max(child_rewards)) if child_rewards else node.reward
        else:
            new_Q = node.reward

        if new_Q == node.Q:
            return  # no change — stop propagation

        node.Q = new_Q
        if node.parent_key is not None and node.parent_key in self._nodes:
            self._propagate_Q(node.parent_key)

    def _ancestry(self, key: Any) -> list[tuple[Any, float]]:
        """Return the path from root to `key` as [(state, reward), ...]."""
        path = []
        cur = key
        while cur is not None:
            node = self._nodes[cur]
            path.append((node.state, node.reward))
            cur = node.parent_key
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    import sys

    failures: list[str] = []

    def check(name: str, cond: bool, msg: str = "") -> None:
        if not cond:
            failures.append(f"FAIL [{name}]: {msg}")
        else:
            print(f"  PASS [{name}]")

    print("=== puct_buffer unit tests ===\n")

    # ------------------------------------------------------------------
    # Basic add / best
    # ------------------------------------------------------------------
    print("-- add / best --")

    buf = PUCTBuffer(c=1.0)
    buf.add("s0", 0.5)
    buf.add("s1", 0.8)
    buf.add("s2", 0.3)

    state, reward = buf.best()
    check("best_returns_max_reward_state", reward == 0.8, f"reward={reward}")
    check("best_returns_correct_state", state == "s1", f"state={state!r}")
    check("len_after_adds", len(buf) == 3, f"len={len(buf)}")

    # Duplicate add is a no-op
    buf.add("s0", 99.0)
    check("duplicate_add_noop", len(buf) == 3, "duplicate changed buffer size")
    check("duplicate_reward_unchanged", buf._nodes[_make_key("s0")].reward == 0.5)

    # ------------------------------------------------------------------
    # Q uses MAX not mean
    # ------------------------------------------------------------------
    print("\n-- Q = MAX not mean --")

    buf2 = PUCTBuffer()
    buf2.add("root", 0.0)
    buf2.add("child_low",  0.1, parent_state="root")
    buf2.add("child_high", 0.9, parent_state="root")

    root_node = buf2._nodes[_make_key("root")]
    check(
        "Q_is_max_not_mean",
        root_node.Q == 0.9,
        f"root.Q={root_node.Q}, expected 0.9 (max), mean would be 0.5",
    )

    # Add another child with even higher reward — Q should update
    buf2.add("child_best", 0.95, parent_state="root")
    check(
        "Q_updates_when_better_child_added",
        root_node.Q == 0.95,
        f"root.Q={root_node.Q}, expected 0.95",
    )

    # ------------------------------------------------------------------
    # Q propagates through grandchildren (MAX of all descendants)
    # ------------------------------------------------------------------
    print("\n-- Q propagation --")

    buf3 = PUCTBuffer()
    buf3.add("r",  0.0)
    buf3.add("c1", 0.3, parent_state="r")
    buf3.add("gc", 0.99, parent_state="c1")   # grandchild

    r_node  = buf3._nodes[_make_key("r")]
    c1_node = buf3._nodes[_make_key("c1")]
    check("grandchild_Q_propagates_to_child",  c1_node.Q == 0.99, f"c1.Q={c1_node.Q}")
    check("grandchild_Q_propagates_to_root",   r_node.Q  == 0.99, f"r.Q={r_node.Q}")

    # Parent with high own reward should NOT lose Q when children underperform
    buf3b = PUCTBuffer()
    buf3b.add("great_parent", 0.9)
    buf3b.add("weak_child",   0.2, parent_state="great_parent")
    gp_node = buf3b._nodes[_make_key("great_parent")]
    check(
        "parent_Q_not_lowered_by_weak_child",
        gp_node.Q == 0.9,
        f"great_parent.Q={gp_node.Q}, expected 0.9 (own reward dominates)",
    )

    # ------------------------------------------------------------------
    # Rank priors: ties get correct average rank (not index-based)
    # ------------------------------------------------------------------
    print("\n-- rank prior tie handling --")

    buf_ties = PUCTBuffer()
    # rewards: s0=0.1 (rank 1), s1=0.5 (tied), s2=0.3 (rank 2), s3=0.5 (tied)
    # After tie-averaging: s0→1, s2→2, s1&s3→(3+4)/2=3.5
    buf_ties.add("s0", 0.1)
    buf_ties.add("s1", 0.5)
    buf_ties.add("s2", 0.3)
    buf_ties.add("s3", 0.5)
    priors_ties = buf_ties._rank_priors()
    p1 = priors_ties[_make_key("s1")]
    p3 = priors_ties[_make_key("s3")]
    p2 = priors_ties[_make_key("s2")]
    check("tied_states_equal_prior", abs(p1 - p3) < 1e-9, f"p1={p1:.6f} p3={p3:.6f}")
    check("tied_states_outrank_lower", p1 > p2, f"tied={p1:.4f} vs s2={p2:.4f}")

    # ------------------------------------------------------------------
    # update() convenience wrapper
    # ------------------------------------------------------------------
    print("\n-- update() --")

    buf4 = PUCTBuffer()
    buf4.add("p", 0.5)
    buf4.update("p", "child_via_update", 0.7)
    check("update_adds_child", len(buf4) == 2, f"len={len(buf4)}")
    check("update_links_child", "child_via_update" in [
        buf4._nodes[ck].state for ck in buf4._nodes[_make_key("p")].children_keys
    ])

    # ------------------------------------------------------------------
    # Exploration: unvisited high-reward states get selected
    # ------------------------------------------------------------------
    print("\n-- exploration: unvisited high-reward states --")

    buf5 = PUCTBuffer(c=1.0)
    # Old state, visited many times
    buf5.add("visited", 0.6)
    buf5._nodes[_make_key("visited")].n = 100
    # New high-reward state, never visited
    buf5.add("fresh_high", 0.9)

    selected = buf5.select(batch_size=2, num_groups=2)
    selected_states = [s for s, _ in selected]
    check(
        "unvisited_high_reward_selected",
        "fresh_high" in selected_states,
        f"selected states: {selected_states}",
    )

    # ------------------------------------------------------------------
    # Exploitation: Q(parent) rises after adding a high-reward child, making
    # the parent score higher than a sibling with no children.
    # We verify PUCT scores directly — not via select() — because select()
    # would correctly pick the child itself (even better warm-start).
    # ------------------------------------------------------------------
    print("\n-- exploitation: high-Q parent outscores peer --")

    buf6 = PUCTBuffer(c=0.01)  # low exploration → scores dominated by Q
    buf6.add("peer_no_children", 0.5)
    buf6.add("parent_explored",  0.5)
    # Give parent_explored a great child: Q should propagate to 0.99
    buf6.add("great_child_2", 0.99, parent_state="parent_explored")

    priors6 = buf6._rank_priors()
    pk_peer   = _make_key("peer_no_children")
    pk_parent = _make_key("parent_explored")
    score_peer   = buf6._puct_score(buf6._nodes[pk_peer],   priors6[pk_peer])
    score_parent = buf6._puct_score(buf6._nodes[pk_parent], priors6[pk_parent])

    check(
        "parent_Q_raised_by_great_child",
        buf6._nodes[pk_parent].Q == 0.99,
        f"parent.Q={buf6._nodes[pk_parent].Q}",
    )
    check(
        "high_Q_parent_outscores_peer",
        score_parent > score_peer,
        f"score_parent={score_parent:.4f}, score_peer={score_peer:.4f}",
    )

    # ------------------------------------------------------------------
    # select() group structure
    # ------------------------------------------------------------------
    print("\n-- select() group structure --")

    buf7 = PUCTBuffer()
    for i in range(10):
        buf7.add(f"s{i}", float(i) / 10)

    result = buf7.select(batch_size=16, num_groups=4)
    check("select_total_length", len(result) == 16, f"len={len(result)}")

    # Each group of 4 should share the same state
    groups_of_4 = [result[i*4:(i+1)*4] for i in range(4)]
    for gi, group in enumerate(groups_of_4):
        states_in_group = [s for s, _ in group]
        check(
            f"group_{gi}_same_state",
            len(set(states_in_group)) == 1,
            f"group {gi} has mixed states: {states_in_group}",
        )

    # Each group should have a DIFFERENT initial state from the others
    group_states = [group[0][0] for group in groups_of_4]
    check(
        "groups_have_distinct_states",
        len(set(group_states)) == 4,
        f"group states: {group_states}",
    )

    # ------------------------------------------------------------------
    # select() raises on batch_size not divisible by num_groups
    # ------------------------------------------------------------------
    print("\n-- select() error handling --")

    buf8 = PUCTBuffer()
    buf8.add("x", 1.0)
    try:
        buf8.select(batch_size=7, num_groups=3)
        check("indivisible_batch_raises", False, "should have raised ValueError")
    except ValueError:
        check("indivisible_batch_raises", True)

    # select() on empty buffer raises
    buf_empty = PUCTBuffer()
    try:
        buf_empty.select(batch_size=4, num_groups=2)
        check("empty_buffer_select_raises", False, "should have raised ValueError")
    except ValueError:
        check("empty_buffer_select_raises", True)

    # ------------------------------------------------------------------
    # Context (ancestry path)
    # ------------------------------------------------------------------
    print("\n-- context / ancestry path --")

    buf9 = PUCTBuffer()
    buf9.add("root",  0.1)
    buf9.add("child", 0.5, parent_state="root")
    buf9.add("grand", 0.9, parent_state="child")

    # Force select to pick "grand" by making it best by far
    buf9._nodes[_make_key("grand")].reward = 10.0
    buf9._propagate_Q(_make_key("child"))
    buf9._propagate_Q(_make_key("root"))

    result9 = buf9.select(batch_size=1, num_groups=1)
    state9, context9 = result9[0]
    check("context_is_list", isinstance(context9, list))
    check(
        "context_starts_at_root",
        context9[0][0] == "root",
        f"context[0]={context9[0]}",
    )
    check(
        "context_ends_at_selected",
        context9[-1][0] == state9,
        f"context[-1]={context9[-1]}, state={state9!r}",
    )
    check(
        "context_length_equals_depth",
        len(context9) == 3,
        f"len={len(context9)}, expected 3",
    )

    # ------------------------------------------------------------------
    # Visit count increments on select
    # ------------------------------------------------------------------
    print("\n-- visit count tracking --")

    buf10 = PUCTBuffer()
    buf10.add("a", 0.5)
    buf10.add("b", 0.6)
    n_before_a = buf10._nodes[_make_key("a")].n
    buf10.select(batch_size=4, num_groups=2)
    T_after = buf10._T
    check("T_incremented_by_num_groups", T_after == 2, f"T={T_after}")
    total_n = sum(n.n for n in buf10._nodes.values())
    check("total_n_equals_T", total_n == T_after, f"sum(n)={total_n}, T={T_after}")

    # ------------------------------------------------------------------
    # numpy array states
    # ------------------------------------------------------------------
    print("\n-- numpy array states --")

    buf11 = PUCTBuffer()
    arr_a = np.array([0.1, 0.5, 0.4])
    arr_b = np.array([0.3, 0.3, 0.4])
    buf11.add(arr_a, 0.7)
    buf11.add(arr_b, 0.9)
    check("numpy_states_len", len(buf11) == 2, f"len={len(buf11)}")
    best_s, best_r = buf11.best()
    check("numpy_best_reward", best_r == 0.9, f"best_r={best_r}")
    check("numpy_best_state", np.array_equal(best_s, arr_b), f"best_s={best_s}")

    # ------------------------------------------------------------------
    print()
    if failures:
        for f in failures:
            print(f)
        print(f"\n{len(failures)} test(s) FAILED")
        import sys; sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    _run_tests()
