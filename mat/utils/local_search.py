import numpy as np


def evaluate_reward_with_current_obs(env_core, flat_action: np.ndarray) -> float:
    """
    Evaluate the reward for a flat action vector without advancing the env state.
    Uses current channel info (history_channel_information) as observation input.

    env_core: underlying MARL env (MarlCustomEnv4)
    flat_action: concatenated [all_UA, all_RB] vector in [0,1]
    """
    act = np.asarray(flat_action, dtype=np.float32).ravel()
    obs = env_core.history_channel_information
    total_rate, _ = env_core.calculate_sumrate(act, obs)
    return float(total_rate)


def local_search_improve(flat_action: np.ndarray,
                         env_core,
                         ua_len: int,
                         budget: int = 20,
                         k: int = 20,
                         early_stop: bool = True,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Simple stochastic local search on UA segment while keeping RB fixed.

    - flat_action: 1D [UA_total, RB_total] in [0,1]
    - env_core: underlying MARL env to evaluate reward without stepping
    - ua_len: total length of UA segment across all agents
    - budget: number of candidate evaluations
    - k: number of UA indices to force to 1 per candidate
    - early_stop: stop immediately on the first improvement
    - rng: optional numpy RNG for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng()

    x0 = np.asarray(flat_action, dtype=np.float32).ravel()
    ua0 = x0[:ua_len].copy()
    rb0 = x0[ua_len:].copy()

    # Baseline reward
    best_score = evaluate_reward_with_current_obs(env_core, x0)
    best_ua = ua0.copy()

    if ua_len <= 0 or budget <= 0 or k <= 0:
        return x0

    k = min(k, ua_len)

    idx_all = np.arange(ua_len)
    for _ in range(budget):
        chosen = rng.choice(idx_all, size=k, replace=False)
        ua_cand = ua0.copy()
        ua_cand[chosen] = 1.0
        x_cand = np.concatenate([ua_cand, rb0])
        score = evaluate_reward_with_current_obs(env_core, x_cand)
        if score > best_score:
            best_score = score
            best_ua = ua_cand
            if early_stop:
                break

    if not np.shares_memory(best_ua, ua0) or not np.array_equal(best_ua, ua0):
        return np.concatenate([best_ua, rb0])
    return x0


def local_search_stepwise(flat_action: np.ndarray,
                          env_core,
                          ua_len: int,
                          include_ua: bool = True,
                          include_rb: bool = True,
                          budget: int | None = None,
                          order: str = "ua_then_rb",
                          ) -> np.ndarray:
    """
    Stepwise greedy local search: set one bit/value to 1, evaluate; if reward improves, keep; then move on.

    - flat_action: 1D vector [UA_total, RB_total] in [0,1]
    - env_core: underlying env to evaluate reward without advancing state
    - ua_len: total UA length
    - include_ua/include_rb: which segments to search
    - budget: optional max number of single-bit evaluations; if None, try all selected indices once
    - order: search order: "ua_then_rb" | "rb_then_ua" | "mixed"
    """
    x = np.asarray(flat_action, dtype=np.float32).ravel().copy()
    n = x.size
    ua_len = int(ua_len)
    ua_len = max(0, min(ua_len, n))
    rb_len = n - ua_len

    # Build candidate index list according to order & segments
    idx_ua = np.arange(0, ua_len) if include_ua and ua_len > 0 else np.array([], dtype=int)
    idx_rb = np.arange(ua_len, n) if include_rb and rb_len > 0 else np.array([], dtype=int)

    if order == "ua_then_rb":
        indices = np.concatenate([idx_ua, idx_rb])
    elif order == "rb_then_ua":
        indices = np.concatenate([idx_rb, idx_ua])
    else:  # mixed: interleave as much as possible
        maxlen = max(idx_ua.size, idx_rb.size)
        roll = []
        for i in range(maxlen):
            if i < idx_ua.size:
                roll.append(idx_ua[i])
            if i < idx_rb.size:
                roll.append(idx_rb[i])
        indices = np.array(roll, dtype=int)

    if indices.size == 0:
        return x

    # Baseline
    best_score = evaluate_reward_with_current_obs(env_core, x)

    # Evaluate each index once (or up to budget)
    evals_left = budget if isinstance(budget, int) and budget is not None and budget > 0 else indices.size
    for idx in indices:
        if evals_left <= 0:
            break
        orig = x[idx]
        if orig >= 1.0:  # already at 1, skip
            evals_left -= 1
            continue
        x[idx] = 1.0
        score = evaluate_reward_with_current_obs(env_core, x)
        if score > best_score:
            best_score = score  # keep the flip
        else:
            x[idx] = orig  # revert
        evals_left -= 1

    return x
