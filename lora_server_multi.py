import socket
import threading
import json
import time
import os
import numpy as np
import torch as T
from argparse import ArgumentParser
from datetime import datetime
from _maddpg import MADDPG, MultiAgentReplayBuffer
import cvxpy as cp


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
G_min = 70.0
G_max = 89.8
G_step = 0.2
GAIN_CANDIDATES = np.round(np.arange(G_min, G_max, G_step), 1)

ACK_TIMEOUT = 30.0
REPORT_TIMEOUT = 90.0
SLOT_PAUSE = 0.5

TX_PORT_BASE = 5555   # TX_i → port 5555 + i
RX_PORT_BASE = 5600   # RX_i → port 5600 + i

FACTOR = 0.95   # tanh clip — must match _maddpg.py
AMPLIFIER = 4      # softmax amplifier (your original line 367)
EPSILON = 1e-20  # numerical safety


# ─────────────────────────────────────────────────────────────────────────────
# actor_dim / critic_dim derived from n_agents
# Mirrors your AC_dims = [17, 4*17] for n_BS=4:
#   actor_dim  = 1+1+1+n + 2+2 + 2*(n-1)
#   critic_dim = n * actor_dim
# ─────────────────────────────────────────────────────────────────────────────
def actor_dim_for(n: int) -> int:
    return 1 + 1 + 1 + n + 2 + 2 + 2 * (n - 1)   # = 3n + 5


def critic_dim_for(n: int) -> int:
    return n * actor_dim_for(n)


# ═════════════════════════════════════════════════════════════════════════════
# MADDPG + Lyapunov Policy  (corrected from uploaded file)
# ═════════════════════════════════════════════════════════════════════════════
class MADDPGPolicy:

    def __init__(self,
                 n_agents=1,
                 n_actions=1,
                 gain_candidates=None,
                 hidden_size=(200, 100, 50),
                 lr_AC=(1e-4, 1e-3),
                 gamma=0.9,
                 tau=0.005,
                 buffer_size=500_000,
                 batch_size=64,
                 action_noise="Gaussian",
                 noise_init=1.0,
                 noise_min=0.05,
                 noise_decay=1 - 5e-5,
                 learn_freq=1,
                 gd_per_slot=1,
                 target_update_freq=1,
                 share_reward=True,
                 load_model=False,
                 chkpt_dir="tmp/maddpg",
                 # Lyapunov parameters
                 V=5000,
                 T_f=0.5,
                 T_b=0.5,
                 T_unit=None,
                 power_max_dBm=40.0,
                 P_avg_dBm=36.13,
                 g_max=1e12,
                 W=1):

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gain_candidates = gain_candidates if gain_candidates is not None \
                                  else GAIN_CANDIDATES
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_freq = learn_freq
        self.gd_per_slot = gd_per_slot
        self.target_update_freq = target_update_freq
        self.share_reward = share_reward
        self.chkpt_dir = chkpt_dir

        # Derived network dimensions
        self.actor_dim = actor_dim_for(n_agents)
        self.critic_dim = critic_dim_for(n_agents)

        # Power limits
        self.P_max_dBm = power_max_dBm
        self.P_max_W = 1e-3 * 10 ** (power_max_dBm / 10.0)  # Watts
        self.P_avg_dBm = P_avg_dBm
        self.P_avg_W = 1e-3 * 10 ** (P_avg_dBm / 10.0)    # Watts

        # Lyapunov parameters
        self.V = V
        self.T_f = T_f
        self.T_b = T_b
        self.T_unit = T_unit if T_unit is not None else T_b
        self.W = W
        self.g_max = g_max
        self.H_constraint = T_f * W * np.log2(1 + g_max * self.P_max_W)

        # Lyapunov queues — one per agent (mirrors your Z_i, H_ji)
        self._Z_i = np.zeros(n_agents, dtype=np.float64)
        self._H_ji = np.zeros(n_agents, dtype=np.float64)
        self._X_ji = np.zeros(n_agents, dtype=np.float64)

        # Per-slot accumulators reset each slot
        self._reward_frame = np.zeros(n_agents, dtype=np.float64)
        self._ratios = np.ones(n_agents) / n_agents

        # Previous-slot measurements for g(t+1) approximation in obs
        self._prev_rssi_dbm = np.full(n_agents, -100.0)
        self._prev_snr_db = np.full(n_agents, 0.0)
        # self._prev_noise_dbfs = np.full(n_agents,  -60.0)

        # Threading
        self._lock = threading.Lock()
        self._step = 0
        self._history = []

        # Initial obs/state
        self._obs_list = [np.zeros(self.actor_dim, dtype=np.float32)
                          for _ in range(n_agents)]
        self._state = np.zeros(self.critic_dim, dtype=np.float32)

        # ── Create MADDPG agents and replay buffer ────────────────────────────
        T.manual_seed(42)
        os.makedirs(chkpt_dir, exist_ok=True)

        self.maddpg_agents = MADDPG(
            actor_dims=[self.actor_dim] * n_agents,
            critic_dims=self.critic_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            fc1=hidden_size[0], fc2=hidden_size[1], fc3=hidden_size[2],
            alpha=lr_AC[0], beta=lr_AC[1],
            gamma=gamma, tau=tau,
            action_noise=action_noise,
            noise_init=noise_init,
            noise_min=noise_min,
            noise_decay=noise_decay,
            chkpt_dir=chkpt_dir)

        if load_model and len(os.listdir(chkpt_dir)) > 1:
            self.maddpg_agents.load_checkpoint()

        self.memory = MultiAgentReplayBuffer(
            max_size=buffer_size,
            critic_dims=self.critic_dim,
            actor_dims=[self.actor_dim] * n_agents,
            n_actions=n_actions,
            n_agents=n_agents,
            batch_size=batch_size)

        print(f"[POLICY] MADDPG + Lyapunov  n_agents={n_agents}")
        print(f"  actor_dim={self.actor_dim}  critic_dim={self.critic_dim}")
        print(f"  V={V}  T_f={T_f}  T_b={T_b}  T_unit={self.T_unit}")
        print(f"  P_max={power_max_dBm} dBm = {self.P_max_W:.4f} W")
        print(f"  P_avg={P_avg_dBm} dBm = {self.P_avg_W:.4f} W")
        print(f"  H_constraint={self.H_constraint:.4f}")
        print(f"  gain range=[{self.gain_candidates[0]}, "
              f"{self.gain_candidates[-1]}] dB  "
              f"({len(self.gain_candidates)} levels)")

    # ─────────────────────────────────────────────────────────────────────────
    # Auxiliary: convert simulated normalised power → nearest dB gain level
    # BUG FIX: original mixed dBm and Watts units — fixed to Watts throughout
    # ─────────────────────────────────────────────────────────────────────────
    def _sim_power_to_gain_dB(self, p_watts: float) -> float:
        """
        Convert a power in Watts (from MADDPG actor output) to the nearest
        dB gain level from GAIN_CANDIDATES.

        The actor outputs a tanh value in [-FACTOR, FACTOR].
        Caller converts: p_watts = P_max_W * (tanh_out + FACTOR) / (2*FACTOR)
        This function snaps p_watts to the nearest discrete gain level.
        """
        p_watts = float(np.clip(p_watts, 0.0, self.P_max_W))
        if p_watts <= 0:
            return float(self.gain_candidates[0])
        p_dBm = 10.0 * np.log10(p_watts * 1e3)   # Watts → dBm
        # Snap to the nearest gain candidate (gain_candidates are in dB = dBm for 1mW ref)
        return float(min(self.gain_candidates,
                         key=lambda g: abs(g - p_dBm)))

    # ─────────────────────────────────────────────────────────────────────────
    # HOOK 1 — choose_action
    # BUG FIX: self.maddpg_agents[i] → self.maddpg_agents (single MADDPG obj)
    #          undefined bare 'i' removed; conversion uses Watts consistently
    # ─────────────────────────────────────────────────────────────────────────
    def choose_action(self, obs_list: list) -> list:
        """
        Mirrors: actions = MADDPG_agents.choose_action(np.array(obs_list))
        Returns list of n_agents gain values in dB.
        """
        # MADDPG_agents.choose_action returns a list of per-agent tensors/arrays
        # each in [-FACTOR, FACTOR] (tanh output)
        raw_actions = self.maddpg_agents.choose_action(
            np.array(obs_list, dtype=np.float32))

        powers_watt = []
        for i in range(self.n_agents):
            tanh_val = float(raw_actions[i].item()
                             if hasattr(raw_actions[i], 'item')
                             else raw_actions[i])
            # Convert tanh → power in Watts (mirrors your lines 251-252)
            p_watts = self.P_max_W * (tanh_val + FACTOR) / (2.0 * FACTOR)
            # gain_dB = self._sim_power_to_gain_dB(p_watts)
            powers_watt.append(p_watts)

        with self._lock:
            step = self._step
        # print(f"[POLICY] step={step}  choose_action → {gains} dB")
        return powers_watt

    # ─────────────────────────────────────────────────────────────────────────
    # HOOK 2 — Reward computation
    # BUG FIX: snr_db must be per-agent (reports is a list);
    #          action_power_watt must be indexed [i];
    #          Lyapunov alpha/beta weighting added (exact match to paper)
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_rewards(self, reports: list, action_watts: list) -> list:
        """
        Args:
            reports     : list of n_agents report dicts from RX nodes
            action_gains: list of n_agents gain values in dB

        Returns rwd_list (shared or individual based on share_reward flag).
        Also stores self._ratios and self._reward_frame for obs builder.

        Reward formula (exact match to your paper lines 336-348):
            alpha = (H_ji[i] / max(max(H_ji), eps)) * T_b
            beta  = (Z_i[i]  / max(max(Z_i),  eps)) * T_b
            SINR  ≈ 10^(avg_snr_db[i] / 10)
            X_ji += T_unit * W * log2(1 + SINR)
            reward[i] = alpha * W * log2(1+SINR) - beta * power_watts[i]
        """
        eps = EPSILON
        h_max = max(float(np.max(self._H_ji)), eps)
        z_max = max(float(np.max(self._Z_i)),  eps)

        rewards = np.zeros(self.n_agents, dtype=np.float64)
        self._X_ji = np.zeros(self.n_agents, dtype=np.float64)  # reset accumulator

        for i in range(self.n_agents):
            # Per-agent SNR from its own RX report
            snr_db_i = float(reports[i].get('avg_snr_db', 0.0))
            sinr_lin = 10 ** (snr_db_i / 10.0)

            # Lyapunov alpha / beta (your lines 336-337)
            alpha = (self._H_ji[i] / h_max) * self.T_b
            beta = (self._Z_i[i] / z_max) * self.T_b

            # Throughput accumulation (your line 347)
            self._X_ji[i] = self.T_unit * self.W * np.log2(1 + sinr_lin)

            # Reward (your line 348)
            rewards[i] = (alpha * self.W * np.log2(1 + sinr_lin)
                          - beta * action_watts[i])

        # Softmax ratios (your lines 367-370)
        import torch
        rwd_t = torch.tensor(rewards.astype(np.float32))
        max_rwd = max(float(rwd_t.max()), eps)
        norm_t = torch.tensor(AMPLIFIER * rewards.astype(np.float32) / max_rwd)
        ratios = torch.softmax(norm_t, dim=0).numpy()

        self._ratios = ratios
        self._reward_frame = rewards.copy()

        # Shared / individual reward (your line 455)
        if self.share_reward:
            team = float(np.sum(rewards))
            rwd_list = [team] * self.n_agents
        else:
            rwd_list = rewards.tolist()

        return rwd_list

    # ─────────────────────────────────────────────────────────────────────────
    # HOOK 3 — Observation builder
    # BUG FIX: now takes list of reports (one per agent); builds per-agent obs
    # ─────────────────────────────────────────────────────────────────────────
    def _build_next_obs(self, reports: list, actions: list,
                        rwd_list: list) -> tuple:
        """
        Builds obs_list_ and state_ from all agents' RX reports.

        Physical → simulated mapping:
          avg_rssi_dbm(t)   / rssi_ref  → Direct_channel g(t)     [line 415]
          avg_rssi_dbm(t-1) / rssi_ref  → Direct_channel g(t+1)   [line 416]
          noise_floor(t)    / nf_ref    → total interf (t)   scaled [line 432]
          noise_floor(t-1)  / nf_ref    → total interf (t+1) scaled [line 434]
          peer_j rssi(t)    / rssi_ref  → Indiv_interf[i,j,t]     [line 440]
          peer_j rssi(t-1)  / rssi_ref  → Indiv_interf_[i,j,t]    [line 442]
        """
        # rssi_ref = 100.0
        # nf_ref = 100.0
        p_ref = float(max(self.gain_candidates))

        rssi_now = np.array([float(r.get('avg_rssi_dbm', -100.0))  # XY: -100 is default, return default if value not exist
                               for r in reports])
        snr_db = np.array([float(r.get('avg_snr_db', 0.0)) for r in reports])

        obs_list_ = []
        for i in range(self.n_agents):
            lo = []

            # [0] normalised TX gain (mirrors power_profile_frame[i])
            lo.append(actions[i])

            # [1] softmax ratio_i
            lo.append(float(self._ratios[i]))

            # [2] sum of all rewards
            lo.append(float(np.sum(self._reward_frame)))

            # [3..3+n-1] per-agent rewards
            lo += self._reward_frame.tolist()

            # [3+n] g(t) — current rssi
            lo.append(rssi_now[i])

            # [4+n] g(t+1) — previous slot rssi as next estimate
            lo.append(self._prev_rssi_dbm[i])

            # [5+n] total interf(t)
            lo.append(rssi_now[i] - snr_db[i])

            # [6+n] total interf(t+1)
            lo.append(self._prev_rssi_dbm[i] - self._prev_snr_db[i])

            # [7+n..] component interf: peer j rssi(t) and rssi(t-1) for j≠i
            peers = [j for j in range(self.n_agents) if j != i]
            for j in peers:
                lo.append(rssi_now[j] - snr_db[j])
                lo.append(self._prev_rssi_dbm[j] - self._prev_snr_db[j])

            obs_list_.append(np.array(lo, dtype=np.float32))

        # state_ = obs_list_to_state_vector(obs_list_)
        state_ = np.concatenate(obs_list_).astype(np.float32)

        # Advance previous-slot buffers
        self._prev_rssi_dbm = rssi_now.copy()
        self._prev_snr_db = snr_db.copy()
        # self._prev_noise_dbfs = noise_now.copy()

        return obs_list_, state_

    # ─────────────────────────────────────────────────────────────────────────
    # HOOK 4 — store_transition (no change needed, uses real memory)
    # ─────────────────────────────────────────────────────────────────────────
    def store_transition(self, obs_list, state, actions, rewards,
                         obs_list_, state_, done):
        self.memory.store_transition(obs_list, state, actions, rewards,
                                     obs_list_, state_, done)

    # ─────────────────────────────────────────────────────────────────────────
    # HOOK 5 — learn (no change needed)
    # ─────────────────────────────────────────────────────────────────────────
    def learn(self, slot_idx: int):
        if slot_idx % self.learn_freq == 0:
            for _ in range(self.gd_per_slot):
                self.maddpg_agents.learn(
                    self.memory,
                    slot_idx=slot_idx,
                    target_freq=self.target_update_freq)

    # ─────────────────────────────────────────────────────────────────────────
    # HOOK 6 — Lyapunov queue update
    # BUG FIX: bare n_agents / V → self.n_agents / self.V
    # ─────────────────────────────────────────────────────────────────────────
    def _lyapunov_update(self, action_watt: list):
        """
        CVXPY solve for x_optimal then queue update.
        Exact match to your lines 481-495.
        """
        n = self.n_agents

        # ── CVXPY solve (your lines 481-486) ─────────────────────────────────
        try:
            x = cp.Variable(n)
            objective = cp.Minimize(
                -self.V * cp.sum(cp.log(x) / np.log(2))
                + cp.sum(self._H_ji @ x)
            )
            constraints = [x <= self.H_constraint, x >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            if prob.status not in ("optimal", "optimal_inaccurate") \
                    or x.value is None:
                raise ValueError(f"CVXPY: {prob.status}")
            x_optimal = np.array(x.value, dtype=np.float64)
        except Exception as e:
            # Closed-form fallback: x* = V / (H_ji * ln2), clipped
            print(f"[LYAPUNOV] CVXPY failed ({e}) — closed-form fallback")
            with np.errstate(divide='ignore', invalid='ignore'):
                x_optimal = np.where(
                    self._H_ji > EPSILON,
                    np.minimum(self.V / (self._H_ji * np.log(2)),
                               self.H_constraint),
                    self.H_constraint
                )

        # ── Queue update (your lines 493-495) ─────────────────────────────────
        for i in range(n):
            self._Z_i[i] = max(self._Z_i[i] + self.T_f * action_watt[i]
                                - self.T_f * self.P_avg_W,   0.0)
            self._H_ji[i] = max(self._H_ji[i] + x_optimal[i]
                                - self._X_ji[i],             0.0)

        print(f"[LYAPUNOV] x_opt={np.round(x_optimal, 4).tolist()}")
        print(f"[LYAPUNOV] Z_i  ={np.round(self._Z_i,  4).tolist()}")
        print(f"[LYAPUNOV] H_ji ={np.round(self._H_ji, 4).tolist()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Main per-slot step — called by orchestrator
    # BUG FIX: removed reference to self._placeholder_buffer
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, obs_list, state, actions, rewards,
             obs_list_, state_, done: bool, slot_idx: int):
        with self._lock:
            self._step += 1
            step = self._step

        self.store_transition(obs_list, state, actions, rewards,
                              obs_list_, state_, done)
        self.learn(slot_idx)
        self._lyapunov_update(actions)

        # Logging
        avg_rwd = float(np.mean(rewards))
        with self._lock:
            self._history.append({
                "step": step,
                "slot": slot_idx,
                "actions": actions,
                "rewards": rewards,
                "Z_i": self._Z_i.tolist(),
                "H_ji": self._H_ji.tolist(),
                "X_ji": self._X_ji.tolist(),
                "buffer": self.memory.mem_cntr
                             if hasattr(self.memory, 'mem_cntr') else -1,
                "timestamp": datetime.now().isoformat(),
            })

        print(f"[MADDPG] step={step:04d}  slot={slot_idx}  "
              f"gains={actions}  avg_rwd={avg_rwd:+.4f}")

        if step % 10 == 0:
            with self._lock:
                last50 = self._history[-min(len(self._history), 50):]
            m = float(np.mean([np.mean(h['rewards']) for h in last50]))
            print(f"[MADDPG] --- avg_rwd(last50)={m:.4f}  "
                  f"Z_i={np.round(self._Z_i,3).tolist()}  "
                  f"H_ji={np.round(self._H_ji,3).tolist()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint / history
    # BUG FIX: now calls real maddpg_agents methods
    # ─────────────────────────────────────────────────────────────────────────
    def save_checkpoint(self):
        self.maddpg_agents.save_checkpoint()
        self.save_history()

    def load_checkpoint(self):
        self.maddpg_agents.load_checkpoint()

    def get_history(self):
        with self._lock:
            return list(self._history)

    def save_history(self, path="maddpg_lora_history.json"):
        with self._lock:
            hist = list(self._history)
        with open(path, 'w') as f:
            json.dump(hist, f, indent=2)
        print(f"[POLICY] History → {path}  ({len(hist)} steps)")


# ═════════════════════════════════════════════════════════════════════════════
# NodeConnection — persistent TCP server socket for one client
# ═════════════════════════════════════════════════════════════════════════════
class NodeConnection:

    def __init__(self, name, port, host='0.0.0.0'):
        self.name = name
        self.port = port
        self.host = host
        self._conn = None
        self._conn_lock = threading.Lock()
        self._running = False
        self._inbox = []
        self._inbox_lock = threading.Lock()
        self._inbox_ev = threading.Event()

    def start(self):
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def stop(self):
        self._running = False
        with self._conn_lock:
            if self._conn:
                try:
                    self._conn.close()
                except:
                    pass

    def is_connected(self) -> bool:
        with self._conn_lock:
            return self._conn is not None

    def send(self, msg: dict):
        with self._conn_lock:
            conn = self._conn
        if conn is None:
            raise OSError(f"[{self.name}] not connected")
        conn.sendall((json.dumps(msg) + '\n').encode('utf-8'))

    def wait_for_message(self, key: str, expected_value=None,
                         timeout: float = ACK_TIMEOUT) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._inbox_lock:
                for i, msg in enumerate(self._inbox):
                    if key in msg and (expected_value is None or
                                       msg[key] == expected_value):
                        self._inbox.pop(i)
                        return msg
            self._inbox_ev.wait(timeout=min(0.1, deadline - time.monotonic()))
            self._inbox_ev.clear()
        raise TimeoutError(
            f"[{self.name}] timeout [{key!r}]={expected_value!r}")

    def _accept_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        srv.settimeout(1.0)
        print(f"[{self.name}] Listening on :{self.port}")
        while self._running:
            try:
                conn, addr = srv.accept()
                print(f"[{self.name}] Connected from {addr}")
                with self._conn_lock:
                    self._conn = conn
                self._recv_loop(conn)
            except socket.timeout:
                continue
            except OSError:
                break

    def _recv_loop(self, conn):
        buf = ""
        while self._running:
            try:
                data = conn.recv(4096).decode('utf-8', errors='replace')
                if not data:
                    break
                buf += data
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        with self._inbox_lock:
                            self._inbox.append(msg)
                        self._inbox_ev.set()
                        k = (msg.get('status') or msg.get('type')
                             or msg.get('cmd') or '?')
                        print(f"[← {self.name}] {k}  "
                              f"agent={msg.get('agent_id','?')}")
                    except json.JSONDecodeError:
                        print(f"[{self.name}] bad JSON: {line[:60]}")
            except OSError:
                break
        print(f"[{self.name}] disconnected")
        with self._conn_lock:
            self._conn = None


# ═════════════════════════════════════════════════════════════════════════════
# Training orchestrator — multi-agent, RX-first, all phases parallel
# BUG FIX: was single TX/RX; now lists of NodeConnection for n_agents
# ═════════════════════════════════════════════════════════════════════════════
class TrainingOrchestrator:

    def __init__(self, tx_nodes: list, rx_nodes: list,
                 policy: MADDPGPolicy,
                 n_slots: int, n_packets: int,
                 period_ms: int = 1000):
        self.tx_nodes = tx_nodes    # list[NodeConnection], len = n_agents
        self.rx_nodes = rx_nodes    # list[NodeConnection], len = n_agents
        self.policy = policy
        self.n_slots = n_slots
        self.n_packets = n_packets
        self.period_ms = period_ms  # ms between packets — passed to TX and RX
        self.n_agents = len(tx_nodes)
        self._stop = threading.Event()
        self._thread = None

    def start_async(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="TrainingLoop")
        self._thread.start()

    def stop(self): self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _all_connected(self) -> bool:
        return (all(n.is_connected() for n in self.tx_nodes) and
                all(n.is_connected() for n in self.rx_nodes))

    def _loop(self):
        n = self.n_agents
        print(f"\n{'═'*66}")
        print(f"  MADDPG + Lyapunov  n_agents={n}  "
              f"slots={self.n_slots}  pkts/slot={self.n_packets}  "
              f"period_ms={self.period_ms}")
        print(f"  actor_dim={self.policy.actor_dim}  "
              f"critic_dim={self.policy.critic_dim}")
        print(f"{'═'*66}\n")

        print("[ORCH] Waiting for all TX and RX nodes ...")
        while not self._stop.is_set():
            if self._all_connected():
                break
            cx = sum(nd.is_connected() for nd in self.tx_nodes)
            cr = sum(nd.is_connected() for nd in self.rx_nodes)
            print(f"[ORCH] TX:{cx}/{n}  RX:{cr}/{n}", end='\r')
            time.sleep(0.5)
        if self._stop.is_set():
            return
        print(f"\n[ORCH] All {n} TX + {n} RX connected — starting\n")
        time.sleep(0.5)

        completed = 0
        for t in range(self.n_slots):
            if self._stop.is_set():
                print("[ORCH] Stop requested")
                break
            ok = self._run_slot(t)
            if ok:
                completed += 1
            if t < self.n_slots - 1 and not self._stop.is_set():
                time.sleep(SLOT_PAUSE)

        stop_cmd = {"cmd": "stop", "timestamp": datetime.now().isoformat()}
        for i in range(self.n_agents):
            for nd, lbl in [(self.tx_nodes[i], f"TX{i}"),
                            (self.rx_nodes[i],  f"RX{i}")]:
                try:
                    nd.send(stop_cmd)
                    print(f"[ORCH → {lbl}] stop")
                except OSError:
                    pass

        print(f"\n{'═'*66}")
        print(f"[ORCH] Done — {completed}/{self.n_slots} slots")
        print(f"{'═'*66}\n")
        self.policy.save_checkpoint()

    def _run_slot(self, t: int) -> bool:
        done = (t == self.n_slots - 1)
        ts = datetime.now().isoformat()
        n = self.n_agents

        print(f"\n{'─'*66}")
        print(f"[ORCH] Slot {t+1}/{self.n_slots}" + ("  [FINAL]" if done else ""))
        print(f"{'─'*66}")

        # ── 1. Policy chooses one gain per agent ──────────────────────────────
        # Change 3: For slot 0 all transmitters start at the maximum gain
        # (89.5 dB) so the first observation is collected at full power.
        # From slot 1 onward the MADDPG policy controls the gain.
        obs_list = self.policy._obs_list
        state    = self.policy._state
        if t == 0:
            actions = [self.policy.P_max_W] * n
            # print(f"[ORCH] Slot 0 — using initial gain {actions[0]} dB "
            #       f"for all {n} agents")
        else:
            actions = self.policy.choose_action(obs_list)  # list[n] in dB

        # ── PHASE A: send start to ALL RX FIRST ──────────────────────────────
        # period_ms tells RX how long to wait before timing out collection
        print("[ORCH] Phase A — start → all RX")
        for i, rx in enumerate(self.rx_nodes):
            try:
                rx.send({"cmd": "start", "n_packets": self.n_packets,
                         "period_ms": self.period_ms,
                         "n_agents": self.n_agents,
                         "slot": t, "agent_id": i, "timestamp": ts})
                print(f"[ORCH → RX{i}] start  "
                      f"n_packets={self.n_packets}  period_ms={self.period_ms}")
            except OSError as e:
                print(f"[ORCH] RX{i} send error: {e} — skipping slot {t+1}")
                return False

        for i, rx in enumerate(self.rx_nodes):
            try:
                rx.wait_for_message("status", "ready", timeout=ACK_TIMEOUT)
                print(f"[ORCH] RX{i} ready")
            except TimeoutError:
                print(f"[ORCH] RX{i} ready timeout — skipping slot {t+1}")
                return False

        # ── PHASE B: send gain to ALL TX ──────────────────────────────────────
        # Change 4: start_delay_ms = 0 for all agents — all TX transmit
        # immediately after receiving the command. Agent identification is
        # handled entirely by the AGT: payload prefix parsed at RX.
        # No time-staggering is applied; simultaneous transmission is accepted
        # (capture effect or collision resolved at the LoRa physical layer).
        print("[ORCH] Phase B — gain → all TX  [start_delay=0 for all]")
        action_gains = []
        for i, tx in enumerate(self.tx_nodes):
            action_gain = G_min + int((actions[i] / self.policy.P_max_W) * (G_max - G_min))
            action_gains.append(action_gain)
            try:
                tx.send({"cmd": "set_gain", "gain": action_gain,
                         "n_packets": self.n_packets,
                         "period_ms": self.period_ms,
                         "start_delay_ms": 0,
                         "slot": t, "agent_id": i, "timestamp": ts})
                print(f"Slot {t} [ORCH → TX{i}] set_gain={action_gain} dB  "
                      f"period_ms={self.period_ms}  start_delay_ms=0")
            except OSError as e:
                print(f"[ORCH] TX{i} send error: {e} — skipping slot {t+1}")
                return False

        for i, tx in enumerate(self.tx_nodes):
            try:
                tx.wait_for_message("status", "ready", timeout=ACK_TIMEOUT)
                print(f"[ORCH] TX{i} ready")
            except TimeoutError:
                print(f"[ORCH] TX{i} ready timeout — skipping slot {t+1}")
                return False

        print(f"[ORCH] All {n} TX + {n} RX ready — slot {t+1} active")

        # ── PHASE C: wait for burst_done from all TX ──────────────────────────
        # Timeout = single burst duration + margin.
        # With start_delay_ms=0 all TX fire simultaneously so the burst
        # window is n_packets * period_ms regardless of n_agents.
        burst_to = self.n_packets * (self.period_ms / 1000.0) + 30.0
        n_sent = {}   # agent_id → actual n_sent reported by TX (for PER)
        for i, tx in enumerate(self.tx_nodes):
            try:
                dm = tx.wait_for_message("status", "burst_done",
                                         timeout=burst_to)
                n_sent[i] = int(dm.get('n_sent', self.n_packets))
                print(f"[ORCH] TX{i} burst_done  "
                      f"n_sent={n_sent[i]}/{self.n_packets}")
            except TimeoutError:
                n_sent[i] = self.n_packets   # assume all sent
                print(f"[ORCH] TX{i} burst_done timeout — "
                      f"assuming n_sent={self.n_packets}")

        # ── PHASE C+: release all RX collectors immediately ───────────────────
        # Sent right after all TX burst_done are collected.
        # This unblocks any RX that received zero matching packets — they would
        # otherwise block in wait_complete() for the full 60s timeout before
        # reporting. With 'release' they respond within milliseconds.
        print("[ORCH] Phase C+ — release → all RX")
        for i, rx in enumerate(self.rx_nodes):
            try:
                rx.send({"cmd": "release", "slot": t,
                         "timestamp": datetime.now().isoformat()})
                print(f"[ORCH → RX{i}] release")
            except OSError as e:
                print(f"[ORCH] RX{i} release send error: {e}")

        # ── PHASE D: wait for report from all RX ──────────────────────────────
        print("[ORCH] Phase D — waiting for all RX reports ...")
        reports = []
        for i, rx in enumerate(self.rx_nodes):
            try:
                rep = rx.wait_for_message("type", "report",
                                          timeout=REPORT_TIMEOUT)
                rep['agent_id']     = i
                rep['current_gain'] = action_gains[i]
                rep['n_sent']       = n_sent.get(i, self.n_packets)
                reports.append(rep)
                n_rx  = rep.get('n_packets', 0)
                n_tx  = rep.get('n_sent',    self.n_packets)
                per   = rep.get('per',  (n_tx - n_rx) / max(n_tx, 1))
                gput  = rep.get('goodput', n_rx / max(n_tx, 1))
                print(f"[ORCH] RX{i} report  "
                      f"rx={n_rx}/{n_tx}  PER={per:.3f}  goodput={gput:.3f}  "
                      f"rssi={rep.get('avg_rssi_dbm','?')} dBm  "
                      f"snr={rep.get('avg_snr_db','?')} dB")
            except TimeoutError:
                print(f"[ORCH] RX{i} report timeout — inserting null report")
                reports.append({
                    "type": "report", "agent_id": i,
                    "n_packets": 0, "n_expected": self.n_packets,
                    "n_sent": n_sent.get(i, self.n_packets),
                    "per": 1.0, "goodput": 0.0,
                    "avg_rssi_dbm": -100.0, "avg_snr_db": 0.0,
                    "noise_floor_dbfs": -60.0, "current_gain": action_gains[i],
                })

        # ── PHASE E: joint MADDPG step ────────────────────────────────────────
        rwd_list = self.policy._compute_rewards(reports, actions)

        obs_list_, state_ = self.policy._build_next_obs(
            reports, actions, rwd_list)

        print(f"\n[ORCH] Slot {t+1} summary:")
        for i in range(n):
            n_rx  = reports[i].get('n_packets', 0)
            n_tx  = reports[i].get('n_sent',    self.n_packets)
            per   = reports[i].get('per',   (n_tx - n_rx) / max(n_tx, 1))
            gput  = reports[i].get('goodput', n_rx / max(n_tx, 1))
            print(f"  Agent {i}: gain={action_gains[i]:5.1f} dB  "
                  f"rx={n_rx}/{n_tx}  PER={per:.3f}  goodput={gput:.3f}  "
                  f"rssi={reports[i].get('avg_rssi_dbm','?'):>8}  "
                  f"snr={reports[i].get('avg_snr_db','?'):>7}  "
                  f"rwd={rwd_list[i]:>+10.4f}")
        print(f"  Z_i ={np.round(self.policy._Z_i,  3).tolist()}")
        print(f"  H_ji={np.round(self.policy._H_ji, 3).tolist()}")

        self.policy.step(
            obs_list=obs_list, state=state,
            actions=actions,   rewards=rwd_list,
            obs_list_=obs_list_, state_=state_,
            done=done, slot_idx=t,
        )

        self.policy._obs_list = obs_list_
        self.policy._state = state_

        return True


# ═════════════════════════════════════════════════════════════════════════════
# Server entry point
# BUG FIX: single tx_node/rx_node → lists; all params wired through
# ═════════════════════════════════════════════════════════════════════════════
class LoRaServer:

    def __init__(self, host='0.0.0.0', n_agents=1,
                 auto_mode=False, n_slots=50, n_packets=10,
                 # MADDPG params
                 n_actions=1,
                 hidden_size=(200, 100, 50), lr_AC=(1e-4, 1e-3),
                 gamma=0.9, tau=0.005,
                 buffer_size=500_000, batch_size=64,
                 action_noise="Gaussian",
                 noise_init=1.0, noise_min=0.05, noise_decay=1-5e-5,
                 learn_freq=1, gd_per_slot=1, target_update_freq=1,
                 share_reward=True, load_model=False,
                 chkpt_dir="tmp/maddpg",
                 # Lyapunov params
                 V=5000, T_f=0.5, T_b=0.5,
                 power_max_dBm=40.0, P_avg_dBm=36.13,
                 g_max=1e12, W=1):

        self.host = host
        self.n_agents = n_agents
        self.auto_mode = auto_mode
        self.n_slots = n_slots
        self.n_packets = n_packets
        self.period_ms = 1000        # ms between packets; override with --period-ms

        # One NodeConnection per agent for TX and RX
        self.tx_nodes = [
            NodeConnection(f"TX{i}", TX_PORT_BASE + i, host)
            for i in range(n_agents)
        ]
        self.rx_nodes = [
            NodeConnection(f"RX{i}", RX_PORT_BASE + i, host)
            for i in range(n_agents)
        ]

        self.policy = MADDPGPolicy(
            n_agents=n_agents,
            n_actions=n_actions,
            gain_candidates=GAIN_CANDIDATES,
            hidden_size=hidden_size,
            lr_AC=lr_AC,
            gamma=gamma, tau=tau,
            buffer_size=buffer_size, batch_size=batch_size,
            action_noise=action_noise,
            noise_init=noise_init, noise_min=noise_min,
            noise_decay=noise_decay,
            learn_freq=learn_freq, gd_per_slot=gd_per_slot,
            target_update_freq=target_update_freq,
            share_reward=share_reward,
            load_model=load_model, chkpt_dir=chkpt_dir,
            V=V, T_f=T_f, T_b=T_b,
            power_max_dBm=power_max_dBm, P_avg_dBm=P_avg_dBm,
            g_max=g_max, W=W,
        )
        self._orch = None

    def start(self):
        for nd in self.tx_nodes + self.rx_nodes:
            nd.start()

        print(f"\n{'═'*66}")
        print(f"  LoRa Server — MADDPG + Lyapunov  (n_agents={self.n_agents})")
        for i in range(self.n_agents):
            print(f"  Agent {i}: TX port {TX_PORT_BASE+i}  "
                  f"RX port {RX_PORT_BASE+i}")
        print(f"  Slots={self.n_slots}  Pkts/slot={self.n_packets}  "
              f"period_ms={self.period_ms}")
        print(f"  actor_dim={self.policy.actor_dim}  "
              f"critic_dim={self.policy.critic_dim}")
        print(f"  Auto={self.auto_mode}")
        print(f"{'═'*66}\n")

        if self.auto_mode:
            self._launch_orch()
        self._interactive()

    def _launch_orch(self):
        self._orch = TrainingOrchestrator(
            tx_nodes=self.tx_nodes, rx_nodes=self.rx_nodes,
            policy=self.policy,
            n_slots=self.n_slots, n_packets=self.n_packets,
            period_ms=self.period_ms,
        )
        self._orch.start_async()
        print("[SERVER] Training loop started")

    def _interactive(self):
        help_text = (
            "\nCommands:"
            "\n  run              start MADDPG training loop"
            "\n  stop             interrupt training loop"
            "\n  slots <N>        set number of training slots"
            "\n  packets <N>      set packets per slot"
            "\n  period <ms>      set ms between packets (use 4000+ for SF12)"
            "\n  connections      show node connection status"
            "\n  lyapunov         show Lyapunov queue values"
            "\n  status           show latest step"
            "\n  history          show last 20 steps"
            "\n  save             save history to JSON"
            "\n  help             show this help"
            "\n  quit             shutdown\n"
        )
        print(help_text)

        while True:
            try:
                line = input("server> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd == 'quit':
                if self._orch:
                    self._orch.stop()
                break
            elif cmd == 'help':
                print(help_text)
            elif cmd == 'run':
                if self._orch and self._orch.is_running():
                    print("[SERVER] Loop already running")
                else:
                    self._launch_orch()
            elif cmd == 'stop':
                if self._orch:
                    self._orch.stop()
                print("[SERVER] Stop signal sent")
            elif cmd == 'slots' and len(parts) == 2:
                try:
                    self.n_slots = int(parts[1])
                    print(f"n_slots={self.n_slots}")
                except ValueError:
                    print("Invalid")
            elif cmd == 'packets' and len(parts) == 2:
                try:
                    self.n_packets = int(parts[1])
                    print(f"n_packets={self.n_packets}")
                except ValueError:
                    print("Invalid")
            elif cmd == 'period' and len(parts) == 2:
                try:
                    self.period_ms = int(parts[1])
                    print(f"period_ms={self.period_ms} ms")
                except ValueError:
                    print("Invalid")
            elif cmd == 'connections':
                for i in range(self.n_agents):
                    tx = "OK" if self.tx_nodes[i].is_connected() else "--"
                    rx = "OK" if self.rx_nodes[i].is_connected() else "--"
                    print(f"  Agent {i}: TX{i}[{tx}] :{TX_PORT_BASE+i}  "
                          f"RX{i}[{rx}] :{RX_PORT_BASE+i}")
            elif cmd == 'lyapunov':
                p = self.policy
                print(f"  Z_i  : {np.round(p._Z_i,  4).tolist()}")
                print(f"  H_ji : {np.round(p._H_ji, 4).tolist()}")
                print(f"  X_ji : {np.round(p._X_ji, 4).tolist()}")
            elif cmd == 'status':
                hist = self.policy.get_history()
                if hist:
                    h = hist[-1]
                    avg = float(np.mean(h['rewards']))
                    print(f"  step={h['step']}  gains={h['actions']}  "
                          f"avg_rwd={avg:+.4f}  {h['timestamp'][:19]}")
                else:
                    print("[SERVER] No steps yet")
            elif cmd == 'history':
                hist = self.policy.get_history()
                if not hist:
                    print("[SERVER] No history")
                else:
                    print(f"\n{'Step':>5}  {'Actions':>25}  "
                          f"{'AvgRwd':>9}  Time")
                    print("─" * 58)
                    for h in hist[-20:]:
                        avg = float(np.mean(h['rewards']))
                        print(f"{h['step']:>5}  "
                              f"{str(h['actions']):>25}  "
                              f"{avg:>+9.4f}  "
                              f"{h['timestamp'][:19]}")
            elif cmd == 'save':
                self.policy.save_history()
            else:
                print(f"Unknown: '{line}' — type 'help'")

        for nd in self.tx_nodes + self.rx_nodes:
            nd.stop()
        print("[SERVER] Shutdown.")


# ═════════════════════════════════════════════════════════════════════════════
# BUG FIX: argument_parser now exposes all MADDPGPolicy parameters
# ═════════════════════════════════════════════════════════════════════════════
def argument_parser():
    p = ArgumentParser(description="LoRa MADDPG+Lyapunov server")
    p.add_argument("--host",               default="0.0.0.0")
    p.add_argument("--n-agents",           type=int,   default=1,
                   help="Number of TX/RX pairs (default 1)")
    p.add_argument("--auto",               action="store_true",
                   help="Start training loop on launch")
    p.add_argument("--slots",              type=int,   default=50)
    p.add_argument("--n-packets",          type=int,   default=10)
    p.add_argument("--period-ms",          type=int,   default=1000,
                   help="ms between packets within a burst "
                        "(default 1000, use 4000+ for SF12)")
    # MADDPG
    p.add_argument("--batch-size",         type=int,   default=64)
    p.add_argument("--buffer-size",        type=int,   default=500_000)
    p.add_argument("--gamma",              type=float, default=0.9)
    p.add_argument("--tau",                type=float, default=0.005)
    p.add_argument("--lr-actor",           type=float, default=1e-4)
    p.add_argument("--lr-critic",          type=float, default=1e-3)
    p.add_argument("--noise-init",         type=float, default=1.0)
    p.add_argument("--noise-min",          type=float, default=0.05)
    p.add_argument("--noise-decay",        type=float, default=1-5e-5)
    p.add_argument("--learn-freq",         type=int,   default=1)
    p.add_argument("--gd-per-slot",        type=int,   default=1)
    p.add_argument("--target-update-freq", type=int,   default=1)
    p.add_argument("--no-share-reward",    action="store_true")
    p.add_argument("--load-model",         action="store_true")
    p.add_argument("--chkpt-dir",          default="tmp/maddpg")
    # Lyapunov
    p.add_argument("--V",                  type=float, default=5000)
    p.add_argument("--T-f",                type=float, default=0.5)
    p.add_argument("--T-b",                type=float, default=0.5)
    p.add_argument("--power-max-dBm",      type=float, default=40.0)
    p.add_argument("--P-avg-dBm",          type=float, default=36.13)
    p.add_argument("--g-max",              type=float, default=1e12)
    p.add_argument("--W",                  type=float, default=1.0)
    return p


def main():
    opt = argument_parser().parse_args()
    os.makedirs(opt.chkpt_dir, exist_ok=True)
    server = LoRaServer(
        host=opt.host,
        n_agents=opt.n_agents,
        auto_mode=opt.auto,
        n_slots=opt.slots,
        n_packets=opt.n_packets,
        lr_AC=(opt.lr_actor, opt.lr_critic),
        gamma=opt.gamma, tau=opt.tau,
        buffer_size=opt.buffer_size,
        batch_size=opt.batch_size,
        noise_init=opt.noise_init, noise_min=opt.noise_min,
        noise_decay=opt.noise_decay,
        learn_freq=opt.learn_freq, gd_per_slot=opt.gd_per_slot,
        target_update_freq=opt.target_update_freq,
        share_reward=not opt.no_share_reward,
        load_model=opt.load_model, chkpt_dir=opt.chkpt_dir,
        V=opt.V, T_f=opt.T_f, T_b=opt.T_b,
        power_max_dBm=opt.power_max_dBm,
        P_avg_dBm=opt.P_avg_dBm,
        g_max=opt.g_max, W=opt.W,
    )
    server.period_ms = opt.period_ms
    server.start()


if __name__ == '__main__':
    main()
