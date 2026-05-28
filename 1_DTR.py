from __future__ import annotations



"""

1_DTR.py — Loto 7/39 predikcija sa regresorom: DecisionTreeRegressor.

  • Vremenski tačan split (bez shuffle).
  • Multi-label cilj: skor po svakom broju 1..39 → top-7.
  • Feature engineering: lag prozor, rolling frekvencije, gap, statistike prošlog kola.
  • Back-test: hits/7, hit%, ROC AUC (macro), LRAP.
  • Snimanje u TXT.
  • Determinizam: PYTHONHASHSEED, np/random = SEED, n_jobs=1, jedna BLAS nit.

"""


import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd

import time
from datetime import datetime, timedelta
import pytz

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score

try:
    from qiskit_machine_learning.utils import algorithm_globals
    algorithm_globals.random_seed = SEED
except Exception:
    pass

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# Konfiguracija
# ============================================================
CSV_PATH    = "/Users/4c/Desktop/GHQ/data/loto7_4622_k42.csv"
OUT_TXT     = "/Users/4c/Desktop/GHQ/KvantniRegresor/1_DTR_predikcija.txt"
N_MIN, N_MAX = 1, 39
K           = 7
LAG         = 5
WINDOWS     = (20, 50, 100)
BACKTEST_N  = 100


def stamp() -> str:
    return datetime.now(pytz.timezone("Europe/Belgrade")).strftime("%d.%m.%Y_%H.%M.%S")


T0 = time.time()
print()
print("🔁 1_DTR — start ", stamp())
print()


# ============================================================
# 1) Učitavanje CSV-a (bez headera, 7 kolona)
# ============================================================
df = pd.read_csv(CSV_PATH, header=None)
df = df.iloc[:, :K].astype(int)
draws = df.values
N = draws.shape[0]
print(f"✅ CSV učitan: {CSV_PATH}")
print(f"   broj izvlačenja: {N}, brojeva po kolu: {K}")
print()


# ============================================================
# 2) Multi-hot reprezentacija svakog izvlačenja (N, 39)
# ============================================================
def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    M = rows.shape[0]
    out = np.zeros((M, N_MAX), dtype=np.int8)
    for i in range(M):
        for v in rows[i]:
            if N_MIN <= v <= N_MAX:
                out[i, v - 1] = 1
    return out


Y_full = draws_to_multihot(draws)


# ============================================================
# 3) Feature engineering
# ============================================================
def build_features(draws_arr: np.ndarray,
                   y_multi: np.ndarray,
                   lag: int = LAG,
                   windows=WINDOWS) -> np.ndarray:
    n, _ = draws_arr.shape
    feats = []
    for L in range(1, lag + 1):
        shifted = np.zeros_like(draws_arr)
        shifted[L:] = draws_arr[:-L]
        feats.append(shifted)
    lag_block = np.concatenate(feats, axis=1)

    cum = np.cumsum(y_multi, axis=0)
    rolling_blocks = []
    for W in windows:
        rolled = np.zeros_like(cum, dtype=float)
        rolled[1:W + 1] = cum[:W]
        rolled[W + 1:] = cum[W:-1] - cum[:-W - 1]
        rolling_blocks.append(rolled / float(W))
    roll_block = np.concatenate(rolling_blocks, axis=1)

    gap = np.zeros((n, N_MAX), dtype=float)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i in range(n):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in draws_arr[i]:
            last_seen[v - 1] = i

    prev = np.zeros_like(draws_arr)
    prev[1:] = draws_arr[:-1]
    s_sum = prev.sum(axis=1, keepdims=True).astype(float)
    s_odd = (prev % 2 == 1).sum(axis=1, keepdims=True).astype(float)
    s_low = (prev <= 19).sum(axis=1, keepdims=True).astype(float)
    s_rng = (prev.max(axis=1, keepdims=True) - prev.min(axis=1, keepdims=True)).astype(float)
    stat_block = np.concatenate([s_sum, s_odd, s_low, s_rng], axis=1)

    return np.concatenate([lag_block, roll_block, gap, stat_block], axis=1)


X_full = build_features(draws, Y_full)
print(f"✅ Features: X_full.shape = {X_full.shape}, Y_full.shape = {Y_full.shape}")
print()

START = max(LAG, max(WINDOWS))


# ============================================================
# 4) (X, y) parovi
# ============================================================
X_all = X_full[START:N].astype(float)
Y_all = Y_full[START:N].astype(float)
print(f"   trening domen: {X_all.shape[0]} parova")
print()


# ============================================================
# 5) Vremenski split + skaliranje
# ============================================================
n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > 200, "Premalo podataka za back-test."

X_train, Y_train = X_all[:n_train], Y_all[:n_train]
X_back,  Y_back  = X_all[n_train:], Y_all[n_train:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_back_s  = scaler.transform(X_back)

X_next_raw = X_full[N - 1:N].astype(float)
X_next_s   = scaler.transform(X_next_raw)


# ============================================================
# 6) Model: DTR
# ============================================================
print("⚛️ Treniranje DTR ...")
model = DecisionTreeRegressor(random_state=SEED, max_depth=10, min_samples_leaf=4)
model.fit(X_train_s, Y_train)
print("   ✅ DTR treniran.")
print()


# ============================================================
# 7) Top-K iz skorova
# ============================================================
def topk_from_scores(scores_1d: np.ndarray, k: int = K) -> np.ndarray:
    s = np.asarray(scores_1d, dtype=float).copy()
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


# ============================================================
# 8) Back-test
# ============================================================
print(f"🧪 Back-test (poslednjih {BACKTEST_N} izvlačenja):")
scores_back = model.predict(X_back_s)

def avg_hits(scores_2d, Y):
    h = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(Y[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        h += len(true_set & pred_set)
    return h / scores_2d.shape[0]

def safe_auc(Y, scores):
    try:
        return roc_auc_score(Y, scores, average="macro")
    except Exception:
        return float("nan")

def safe_lrap(Y, scores):
    try:
        return label_ranking_average_precision_score(Y.astype(int), scores)
    except Exception:
        return float("nan")

hits = avg_hits(scores_back, Y_back)
auc  = safe_auc(Y_back, scores_back)
lrap = safe_lrap(Y_back, scores_back)

print(f"   {'model':<6} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"   {'DTR':<6} {hits:>8.3f} {100*hits/K:>6.1f}% {auc:>7.3f} {lrap:>7.3f}")
print(f"   (slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


# ============================================================
# 9) Predikcija SLEDEĆEG kola
# ============================================================
next_score = model.predict(X_next_s)[0]
pick = topk_from_scores(next_score)
print(f"🎯 DTR sledeće kolo -> {pick.tolist()}")
print()


# ============================================================
# 10) Validacija + opis + snimanje u TXT
# ============================================================
def describe(p: np.ndarray) -> str:
    s = p.sum()
    odd = int((p % 2 == 1).sum())
    low = int((p <= 19).sum())
    rng = int(p.max() - p.min())
    return f"suma={s}, neparnih={odd}/{K}, niskih(≤19)={low}/{K}, raspon={rng}"

assert len(set(pick.tolist())) == K
assert pick.min() >= N_MIN and pick.max() <= N_MAX
assert list(pick) == sorted(pick.tolist())
print(f"✅ DTR validan ({describe(pick)}).")

elapsed = time.time() - T0
with open(OUT_TXT, "a", encoding="utf-8") as f:
    f.write(f"\n--- {stamp()} (seed={SEED}, N={N}) ---\n")
    f.write(f"DTR -> {pick.tolist()}  ({describe(pick)})\n")
    f.write(f"back-test: hits/7={hits:.3f}, hit%={100*hits/K:.1f}, AUC={auc:.3f}, LRAP={lrap:.3f}\n")
    f.write(f"ukupno_vreme={str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)\n")
print(f"📝 Snimljeno u: {OUT_TXT}")

print()
print("🔁 1_DTR — stop ", stamp())
print(f"⏱️  Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()



"""
🔁 1_DTR — start  28.05.2026_15.00.49

✅ CSV učitan: /loto7_4622_k42.csv
   broj izvlačenja: 4622, brojeva po kolu: 7

✅ Features: X_full.shape = (4622, 195), Y_full.shape = (4622, 39)

   trening domen: 4522 parova

⚛️ Treniranje DTR ...
   ✅ DTR treniran.

🧪 Back-test (poslednjih 100 izvlačenja):
   model    hits/7    hit%     AUC    LRAP
   DTR       1.480   21.1%   0.506   0.254
   (slučajan baseline ≈ 1.256 hits/7)

🎯 DTR sledeće kolo -> [8, 13, 16, 23, 31, 34, 37]

✅ DTR validan (suma=162, neparnih=4/7, niskih(≤19)=3/7, raspon=29).
📝 Snimljeno u: /1_DTR_predikcija.txt

🔁 1_DTR — stop  28.05.2026_15.00.49
⏱️  Ukupno vreme: 0:00:00  (0.3 s)
"""
