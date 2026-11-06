

# Combined 2-key TDES EDE: Boolean/ReLU correctness and synthetic key recovery.
# Original recovery functions retained from the supplied v3 program.
import csv
from pathlib import Path
#!/usr/bin/env python3
"""
tdes_2key_dnn_full_recovery_v2.py

Self-contained research experiment for 2-key TDES EDE:

    C = E_{K1}( D_{K2}( E_{K1}(P) ) )

Pipeline
--------
1. Build two 56-bit external DES keys K1, K2 and insert DES odd parity.
2. Generate real 2-key TDES EDE known plaintext/ciphertext pairs.
3. Synthetic DNN intermediate leakage:
      - recover K1 encryption round-1 subkey  (DES K1, subkeys[0])
      - recover K2 decryption round-1 subkey  (DES K16, subkeys[15])
4. Invert PC2:
      - K1 round subkey -> 8 unknown PC1-output bits; undo round-1 rotation
      - K2 K16 subkey   -> 8 unknown PC1-output bits; no rotation needed
5. Enumerate 2^8 candidate full DES keys for each master key.
6. Under the same synthetic intermediate-state oracle assumption, resolve:
      K1 candidates with P -> E_K1(P)
      K2 candidates with A -> D_K2(A)
7. Validate the recovered pair with complete 2-key TDES EDE PT/CT pairs.

The DNN leakage oracle is synthetic and explicitly exposes natural real-valued
XOR outputs at selected DES round-key mixing nodes. It is intended for
research/self-study, not as a claim about physical leakage of ordinary TDES.

Python 3.8+
NumPy required.
"""

import time
import numpy as np

# ============================================================
# DES tables
# ============================================================

IP = [
58,50,42,34,26,18,10,2, 60,52,44,36,28,20,12,4,
62,54,46,38,30,22,14,6, 64,56,48,40,32,24,16,8,
57,49,41,33,25,17,9,1, 59,51,43,35,27,19,11,3,
61,53,45,37,29,21,13,5, 63,55,47,39,31,23,15,7
]

FP = [
40,8,48,16,56,24,64,32, 39,7,47,15,55,23,63,31,
38,6,46,14,54,22,62,30, 37,5,45,13,53,21,61,29,
36,4,44,12,52,20,60,28, 35,3,43,11,51,19,59,27,
34,2,42,10,50,18,58,26, 33,1,41,9,49,17,57,25
]

E = [
32,1,2,3,4,5, 4,5,6,7,8,9, 8,9,10,11,12,13,
12,13,14,15,16,17, 16,17,18,19,20,21,
20,21,22,23,24,25, 24,25,26,27,28,29,
28,29,30,31,32,1
]

P = [
16,7,20,21,29,12,28,17,
1,15,23,26,5,18,31,10,
2,8,24,14,32,27,3,9,
19,13,30,6,22,11,4,25
]

PC1 = [
57,49,41,33,25,17,9,
1,58,50,42,34,26,18,
10,2,59,51,43,35,27,
19,11,3,60,52,44,36,
63,55,47,39,31,23,15,
7,62,54,46,38,30,22,
14,6,61,53,45,37,29,
21,13,5,28,20,12,4
]

PC2 = [
14,17,11,24,1,5,3,28,
15,6,21,10,23,19,12,4,
26,8,16,7,27,20,13,2,
41,52,31,37,47,55,30,40,
51,45,33,48,44,49,39,56,
34,53,46,42,50,36,29,32
]

SHIFTS = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

SBOX = [
[
[14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
[0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
[4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
[15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]],
[
[15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
[3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
[0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
[13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9]],
[
[10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
[13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
[13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
[1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12]],
[
[7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
[13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
[10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
[3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14]],
[
[2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
[14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
[4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
[11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3]],
[
[12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
[10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
[9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
[4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13]],
[
[4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
[13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
[1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
[6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12]],
[
[13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
[1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
[7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
[2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]]
]

# ============================================================
# Bit helpers
# ============================================================

def permute(bits, table):
    return bits[np.asarray(table, dtype=np.int64) - 1].astype(np.uint8)

def hex64_to_bits(h):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)).astype(np.uint8)

def bits_to_hex(bits):
    return np.packbits(bits.astype(np.uint8)).tobytes().hex().upper()

def bytes_to_bits(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)

def bits_to_bytes(bits):
    return np.packbits(bits.astype(np.uint8)).tobytes()

def external7_to_key64(ext7):
    assert len(ext7) == 7
    b56 = bytes_to_bits(ext7)
    out = np.zeros(64, dtype=np.uint8)
    for i in range(8):
        d = b56[7*i:7*(i+1)]
        parity = 1 ^ (int(np.sum(d)) & 1)
        out[8*i:8*i+7] = d
        out[8*i+7] = parity
    return out

def key64_to_external7(key64):
    b56 = np.concatenate([key64[8*i:8*i+7] for i in range(8)])
    return bits_to_bytes(b56)

# ============================================================
# Boolean DES
# ============================================================

def rol(a, n):
    return np.concatenate((a[n:], a[:n]))

def des_round_keys_48(key64):
    cd = permute(key64, PC1)
    C, D = cd[:28].copy(), cd[28:].copy()
    keys = []
    for s in SHIFTS:
        C, D = rol(C, s), rol(D, s)
        keys.append(permute(np.concatenate((C, D)), PC2))
    return keys

def sbox_layer(x48):
    out = np.zeros(32, dtype=np.uint8)
    for box in range(8):
        b = x48[6*box:6*(box+1)]
        row = (int(b[0]) << 1) | int(b[5])
        col = (int(b[1]) << 3) | (int(b[2]) << 2) | (int(b[3]) << 1) | int(b[4])
        v = SBOX[box][row][col]
        for j in range(4):
            out[4*box+j] = (v >> (3-j)) & 1
    return out

def ffunc(R, K):
    return permute(sbox_layer(permute(R, E) ^ K), P)

def des_encrypt_bits(pt, key64):
    keys = des_round_keys_48(key64)
    st = permute(pt, IP)
    L, R = st[:32].copy(), st[32:].copy()
    for k in keys:
        L, R = R, L ^ ffunc(R, k)
    return permute(np.concatenate((R, L)), FP)

def des_decrypt_bits(ct, key64):
    keys = des_round_keys_48(key64)
    st = permute(ct, IP)
    L, R = st[:32].copy(), st[32:].copy()
    for k in keys[::-1]:
        L, R = R, L ^ ffunc(R, k)
    return permute(np.concatenate((R, L)), FP)

def tdes_2key_encrypt_bits(pt, key1_64, key2_64):
    x = des_encrypt_bits(pt, key1_64)
    x = des_decrypt_bits(x, key2_64)
    x = des_encrypt_bits(x, key1_64)
    return x

def tdes_2key_decrypt_bits(ct, key1_64, key2_64):
    x = des_decrypt_bits(ct, key1_64)
    x = des_encrypt_bits(x, key2_64)
    x = des_decrypt_bits(x, key1_64)
    return x

# ============================================================
# Natural real-valued XOR leakage
# ============================================================

_ACTIVE_XOR = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

def corner_indicator_real(x, a, c):
    match = float(np.sum(1.0 - np.abs(x.astype(float) - a.astype(float))))
    return max((match - x.size + c) / c, 0.0)

def nn_xor2_real(x, y, c):
    xy = np.array([x, y], dtype=np.float64)
    return sum(corner_indicator_real(xy, a, c) for a in _ACTIVE_XOR)

class RoundXorOracle:
    """
    Synthetic oracle for a selected 48-bit DES subkey at the first round
    interface of an encrypt/decrypt stage.

    Given an externally supplied 64-bit stage input X, it exposes the 48
    real XOR-node outputs:
        E(R0) XOR Kround
    where R0 is the right half of IP(X).
    """
    def __init__(self, roundkey48, mode="sym", c=0.5, eps=1e-2):
        self.K = roundkey48.astype(np.uint8)
        self.mode = mode
        self.c = float(c)
        self.eps = float(eps)

    def enc(self, b):
        if self.mode == "sym":
            return float(b)
        return self.eps if b == 0 else 1.0 - self.eps

    def query(self, block64):
        st = permute(block64.astype(np.uint8), IP)
        ER = permute(st[32:], E)
        z = np.zeros(48, dtype=np.float64)
        for i in range(48):
            z[i] = nn_xor2_real(self.enc(int(ER[i])), self.enc(int(self.K[i])), self.c)
        return z

def rpos_for_E_index(i):
    return E[i] - 1

def make_block_from_L0R0(L0, R0):
    return permute(np.concatenate((L0, R0)).astype(np.uint8), FP)

def genpairs(i, rng):
    L0 = rng.integers(0, 2, 32, dtype=np.uint8)
    R0 = rng.integers(0, 2, 32, dtype=np.uint8)
    rp = rpos_for_E_index(i)
    ERi = int(R0[rp])
    R0p = R0.copy()
    R0p[rp] ^= 1
    return make_block_from_L0R0(L0, R0), make_block_from_L0R0(L0, R0p), ERi

def recover_roundkey48(oracle, trials_per_bit=5, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    khat = np.zeros(48, dtype=np.uint8)
    for i in range(48):
        votes = 0
        for _ in range(trials_per_bit):
            a0, a1, ERi = genpairs(i, rng)
            z0, z1 = oracle.query(a0)[i], oracle.query(a1)[i]
            if ERi == 0:
                a, b = z0, z1
            else:
                a, b = z1, z0
            votes += (0 if a < b else 1)
        khat[i] = 1 if votes > trials_per_bit // 2 else 0
    return khat

# ============================================================
# Master-key candidates from a DES round subkey
# ============================================================

def inverse_pc2_partial(roundkey48):
    cd = -np.ones(56, dtype=np.int16)
    for j, p in enumerate(PC2):
        cd[p - 1] = int(roundkey48[j])
    return cd

def undo_round1_rotation(cd1):
    C1, D1 = cd1[:28], cd1[28:]
    return np.concatenate((np.roll(C1, 1), np.roll(D1, 1)))

def round1_subkey_to_CD0_partial(k48):
    return undo_round1_rotation(inverse_pc2_partial(k48))

def round16_subkey_to_CD0_partial(k48):
    # Total DES rotation is 28 positions, so C16=C0 and D16=D0.
    return inverse_pc2_partial(k48)

def key64_from_CD0(cd0):
    key64 = np.zeros(64, dtype=np.uint8)
    for j, p in enumerate(PC1):
        key64[p - 1] = int(cd0[j])

    # DES parity positions are 8,16,...,64.
    for i in range(8):
        data7 = key64[8*i:8*i+7]
        key64[8*i+7] = 1 ^ (int(np.sum(data7)) & 1)
    return key64

def enumerate_key64_candidates(cd0_partial):
    unknown = np.where(cd0_partial < 0)[0].tolist()
    if len(unknown) != 8:
        raise ValueError("Expected exactly 8 unknown PC2 bits, got %d" % len(unknown))
    out = []
    for mask in range(256):
        cd = cd0_partial.copy()
        for j, pos in enumerate(unknown):
            cd[pos] = (mask >> j) & 1
        out.append(key64_from_CD0(cd.astype(np.uint8)))
    return out

# ============================================================
# Candidate refinement using the same intermediate-oracle model
# ============================================================

def tdes_2key_encrypt_trace(pt, key1_64, key2_64):
    """
    Instrumented EDE evaluation:
        A = E_K1(P)
        B = D_K2(A)
        C = E_K1(B)
    Returns (A, B, C).

    The synthetic DNN experiment already assumes intermediate-stage access;
    these values are therefore used to resolve the 8 PC2-missing bits of
    each recovered round key efficiently and consistently.
    """
    A = des_encrypt_bits(pt, key1_64)
    B = des_decrypt_bits(A, key2_64)
    C = des_encrypt_bits(B, key1_64)
    return A, B, C

def filter_k1_candidates_from_stage1(k1_candidates, pt, A_true):
    survivors = []
    for k1 in k1_candidates:
        if np.array_equal(des_encrypt_bits(pt, k1), A_true):
            survivors.append(k1.copy())
    return survivors

def filter_k2_candidates_from_stage2(k2_candidates, A_true, B_true):
    survivors = []
    for k2 in k2_candidates:
        if np.array_equal(des_decrypt_bits(A_true, k2), B_true):
            survivors.append(k2.copy())
    return survivors

# ============================================================
# Self-test / experiment
# ============================================================

def hamming_distance_bits(a, b):
    return int(np.count_nonzero(a.astype(np.uint8) ^ b.astype(np.uint8)))

def random_external7(rng):
    return rng.integers(0, 256, size=7, dtype=np.uint8).tobytes()

def recover_one_pair(ext1, ext2, trial_index, trials_per_bit=5, c=0.5, eps=1e-2):
    """
    Execute one complete synthetic DNN/intermediate-oracle 2-key TDES recovery.

    Returns a dictionary containing:
      - exact recovery flags
      - 48-bit recovered-round-key Hamming distances
      - 56-bit external-master-key Hamming distances
      - candidate counts
      - elapsed recovery time
    """
    t0 = time.perf_counter()

    key1 = external7_to_key64(ext1)
    key2 = external7_to_key64(ext2)

    # Fixed known plaintext. A second PT/CT pair is generated only for
    # final whole-EDE validation.
    pt0 = hex64_to_bits("0000000000000000")
    pt1 = hex64_to_bits("0123456789ABCDEF")

    A0, B0, ct0 = tdes_2key_encrypt_trace(pt0, key1, key2)
    ct1 = tdes_2key_encrypt_bits(pt1, key1, key2)

    # True target subkeys.
    rk1_true = des_round_keys_48(key1)[0]
    rk16_true = des_round_keys_48(key2)[15]

    # Synthetic DNN round-key leakage recovery.
    oracle_k1 = RoundXorOracle(rk1_true, mode="change", c=c, eps=eps)
    oracle_k2 = RoundXorOracle(rk16_true, mode="change", c=c, eps=eps)

    # Distinct deterministic per-trial RNG seeds.
    rk1_hat = recover_roundkey48(
        oracle_k1, trials_per_bit=trials_per_bit,
        rng_seed=2 * trial_index + 1
    )
    rk16_hat = recover_roundkey48(
        oracle_k2, trials_per_bit=trials_per_bit,
        rng_seed=2 * trial_index + 2
    )

    rk1_hd = hamming_distance_bits(rk1_hat, rk1_true)
    rk16_hd = hamming_distance_bits(rk16_hat, rk16_true)

    # If either round subkey contains errors, inverse-PC2 candidate search
    # need not contain the true key. We still measure master-key HD by
    # reporting 56 on unrecoverable trials unless a candidate is found.
    k1_rec = None
    k2_rec = None
    cand1_count = 0
    cand2_count = 0
    surv1_count = 0
    surv2_count = 0

    try:
        cand1 = enumerate_key64_candidates(
            round1_subkey_to_CD0_partial(rk1_hat)
        )
        cand1_count = len(cand1)
        s1 = filter_k1_candidates_from_stage1(cand1, pt0, A0)
        surv1_count = len(s1)
        if len(s1) == 1:
            k1_rec = s1[0]
    except Exception:
        pass

    try:
        cand2 = enumerate_key64_candidates(
            round16_subkey_to_CD0_partial(rk16_hat)
        )
        cand2_count = len(cand2)
        s2 = filter_k2_candidates_from_stage2(cand2, A0, B0)
        surv2_count = len(s2)
        if len(s2) == 1:
            k2_rec = s2[0]
    except Exception:
        pass

    # Master-key HD is measured on the effective external 56-bit keys.
    ext1_bits = bytes_to_bits(ext1)
    ext2_bits = bytes_to_bits(ext2)

    if k1_rec is not None:
        ext1_rec = key64_to_external7(k1_rec)
        k1_master_hd = hamming_distance_bits(
            bytes_to_bits(ext1_rec), ext1_bits
        )
    else:
        ext1_rec = None
        k1_master_hd = 56

    if k2_rec is not None:
        ext2_rec = key64_to_external7(k2_rec)
        k2_master_hd = hamming_distance_bits(
            bytes_to_bits(ext2_rec), ext2_bits
        )
    else:
        ext2_rec = None
        k2_master_hd = 56

    # Complete EDE validation with two known pairs.
    final_valid = False
    if k1_rec is not None and k2_rec is not None:
        final_valid = (
            np.array_equal(
                tdes_2key_encrypt_bits(pt0, k1_rec, k2_rec), ct0
            )
            and np.array_equal(
                tdes_2key_encrypt_bits(pt1, k1_rec, k2_rec), ct1
            )
        )

    k1_exact = (ext1_rec == ext1) if ext1_rec is not None else False
    k2_exact = (ext2_rec == ext2) if ext2_rec is not None else False
    pair_exact = bool(k1_exact and k2_exact and final_valid)

    elapsed = time.perf_counter() - t0

    return {
        "trial": trial_index,
        "k1_round_hd": rk1_hd,
        "k2_round_hd": rk16_hd,
        "k1_master_hd": k1_master_hd,
        "k2_master_hd": k2_master_hd,
        "k1_round_exact": (rk1_hd == 0),
        "k2_round_exact": (rk16_hd == 0),
        "k1_master_exact": bool(k1_exact),
        "k2_master_exact": bool(k2_exact),
        "pair_exact": pair_exact,
        "final_valid": bool(final_valid),
        "k1_candidates": cand1_count,
        "k2_candidates": cand2_count,
        "k1_survivors": surv1_count,
        "k2_survivors": surv2_count,
        "elapsed": elapsed,
    }


def mean(values):
    return float(sum(values)) / float(len(values)) if values else 0.0


def main():
    # ========================================================
    # User-configurable experiment parameters
    # ========================================================
    NUM_TRIALS = 1000
    RANDOM_SEED = 20260810
    TRIALS_PER_BIT = 5
    C = 0.5
    EPS = 1e-2
    PROGRESS_EVERY = 25
    # ========================================================

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    print("=" * 80)
    print("2-key TDES EDE synthetic-DNN key-recovery batch experiment")
    print("C = E_K1(D_K2(E_K1(P)))")
    print("=" * 80)
    print("Random (K1,K2) pairs =", NUM_TRIALS)
    print("Trials per subkey bit =", TRIALS_PER_BIT)
    print("Random seed           =", RANDOM_SEED)
    print("Intermediate-oracle   = enabled (synthetic research model)")
    print("=" * 80)

    experiment_t0 = time.perf_counter()

    for t in range(NUM_TRIALS):
        # Ensure two independently generated 56-bit effective DES keys.
        ext1 = random_external7(rng)
        ext2 = random_external7(rng)

        r = recover_one_pair(
            ext1, ext2,
            trial_index=t,
            trials_per_bit=TRIALS_PER_BIT,
            c=C,
            eps=EPS,
        )
        results.append(r)

        if (
            (t + 1) == 1
            or (t + 1) % PROGRESS_EVERY == 0
            or (t + 1) == NUM_TRIALS
        ):
            pair_success_so_far = sum(x["pair_exact"] for x in results)
            print(
                "trial {:4d}/{:4d}: pair_success={:4d}/{:4d} "
                "last_time={:.4f}s "
                "rkHD=({:d},{:d}) masterHD=({:d},{:d})".format(
                    t + 1, NUM_TRIALS,
                    pair_success_so_far, t + 1,
                    r["elapsed"],
                    r["k1_round_hd"], r["k2_round_hd"],
                    r["k1_master_hd"], r["k2_master_hd"],
                )
            )

    total_elapsed = time.perf_counter() - experiment_t0

    # --------------------------------------------------------
    # Aggregate success statistics
    # --------------------------------------------------------
    k1_round_ok = sum(r["k1_round_exact"] for r in results)
    k2_round_ok = sum(r["k2_round_exact"] for r in results)
    both_round_ok = sum(
        r["k1_round_exact"] and r["k2_round_exact"] for r in results
    )

    k1_master_ok = sum(r["k1_master_exact"] for r in results)
    k2_master_ok = sum(r["k2_master_exact"] for r in results)
    pair_ok = sum(r["pair_exact"] for r in results)

    # Hamming-distance arrays.
    k1_rhd = [r["k1_round_hd"] for r in results]
    k2_rhd = [r["k2_round_hd"] for r in results]
    k1_mhd = [r["k1_master_hd"] for r in results]
    k2_mhd = [r["k2_master_hd"] for r in results]
    times = [r["elapsed"] for r in results]

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print("[Round-subkey recovery: 48 bits each]")
    print(
        "K1 encryption round-1 exact : {:4d}/{:4d} = {:7.3f}%".format(
            k1_round_ok, NUM_TRIALS, 100.0 * k1_round_ok / NUM_TRIALS
        )
    )
    print(
        "K2 decryption K16 exact      : {:4d}/{:4d} = {:7.3f}%".format(
            k2_round_ok, NUM_TRIALS, 100.0 * k2_round_ok / NUM_TRIALS
        )
    )
    print(
        "Both round subkeys exact     : {:4d}/{:4d} = {:7.3f}%".format(
            both_round_ok, NUM_TRIALS, 100.0 * both_round_ok / NUM_TRIALS
        )
    )
    print(
        "Average HD K1 round subkey   = {:.6f} / 48".format(mean(k1_rhd))
    )
    print(
        "Average HD K2 K16 subkey     = {:.6f} / 48".format(mean(k2_rhd))
    )
    print(
        "Maximum HD K1/K2 round keys  = {} / {}".format(
            max(k1_rhd), max(k2_rhd)
        )
    )

    print()
    print("[External DES master-key recovery: 56 bits each]")
    print(
        "K1 master exact              : {:4d}/{:4d} = {:7.3f}%".format(
            k1_master_ok, NUM_TRIALS, 100.0 * k1_master_ok / NUM_TRIALS
        )
    )
    print(
        "K2 master exact              : {:4d}/{:4d} = {:7.3f}%".format(
            k2_master_ok, NUM_TRIALS, 100.0 * k2_master_ok / NUM_TRIALS
        )
    )
    print(
        "Complete (K1,K2) exact       : {:4d}/{:4d} = {:7.3f}%".format(
            pair_ok, NUM_TRIALS, 100.0 * pair_ok / NUM_TRIALS
        )
    )
    print(
        "Average HD K1 master key     = {:.6f} / 56".format(mean(k1_mhd))
    )
    print(
        "Average HD K2 master key     = {:.6f} / 56".format(mean(k2_mhd))
    )
    print(
        "Maximum HD K1/K2 masters     = {} / {}".format(
            max(k1_mhd), max(k2_mhd)
        )
    )

    print()
    print("[Timing]")
    print("Average recovery time / pair = {:.6f} sec".format(mean(times)))
    print("Minimum recovery time / pair = {:.6f} sec".format(min(times)))
    print("Maximum recovery time / pair = {:.6f} sec".format(max(times)))
    print("Total experiment time        = {:.6f} sec".format(total_elapsed))

    # Additional distribution counts useful for paper tables.
    zero_round_hd_pairs = sum(
        (r["k1_round_hd"] == 0 and r["k2_round_hd"] == 0)
        for r in results
    )
    zero_master_hd_pairs = sum(
        (r["k1_master_hd"] == 0 and r["k2_master_hd"] == 0)
        for r in results
    )

    print()
    print("[Consistency]")
    print("Zero-HD round-subkey pairs   =", zero_round_hd_pairs)
    print("Zero-HD master-key pairs     =", zero_master_hd_pairs)

    if pair_ok == NUM_TRIALS:
        print()
        print(
            "SUCCESS: all {} random 2-key TDES pairs were recovered exactly."
            .format(NUM_TRIALS)
        )
    else:
        print()
        print(
            "PARTIAL: {} of {} complete key pairs were recovered exactly."
            .format(pair_ok, NUM_TRIALS)
        )



# ============================================================
# Exact binary-input ReLU corner-sum TDES realization
# ============================================================
# A corner indicator is ReLU(sum(1-|x_i-a_i|)-n+c)/c.
# This implementation evaluates all active S-box corners using vectorized
# NumPy arrays; the XOR and S-box are NOT Boolean lookup shortcuts.

_SBOX_ACTIVE = None

def active_sbox_corners():
    global _SBOX_ACTIVE
    if _SBOX_ACTIVE is None:
        result = []
        for box in range(8):
            active = [[] for _ in range(4)]
            for n in range(64):
                b = np.array([(n >> (5-i)) & 1 for i in range(6)], dtype=np.float64)
                row = (int(b[0]) << 1) | int(b[5])
                col = sum(int(b[j]) << (4-j) for j in range(1,5))
                val = SBOX[box][row][col]
                for j in range(4):
                    if (val >> (3-j)) & 1:
                        active[j].append(b)
            result.append([np.asarray(a, dtype=np.float64) for a in active])
        _SBOX_ACTIVE = result
    return _SBOX_ACTIVE

def relu(x):
    return np.maximum(x, 0.0)

def corner_sum_binary(x, active, c):
    x = np.asarray(x, dtype=np.float64)
    match = np.sum(1.0 - np.abs(active - x), axis=1)
    return float(np.sum(relu((match - x.size + c) / c)))

def dnn_xor_vector(a, b, c):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.stack((a,b),axis=-1)
    matches = np.sum(1.0 - np.abs(z[...,None,:] - _ACTIVE_XOR),axis=-1)
    return np.sum(relu((matches - 2 + c)/c),axis=-1)

def dnn_feistel(R, K, c, corners):
    expanded = permute(R, E)
    x = dnn_xor_vector(expanded, K, c)
    out = np.zeros(32, dtype=np.float64)
    for box in range(8):
        for j in range(4):
            out[4*box+j] = corner_sum_binary(x[6*box:6*(box+1)], corners[box][j], c)
    return permute(out, P)

def dnn_des_crypt(block, key, decrypt=False, c=0.5):
    keys = des_round_keys_48(key)
    if decrypt:
        keys = keys[::-1]
    corners = active_sbox_corners()
    state = permute(block, IP).astype(np.float64)
    L, R = state[:32].copy(), state[32:].copy()
    for k in keys:
        f = dnn_feistel(R, k, c, corners)
        L, R = R, dnn_xor_vector(L, f, c)
    return permute(np.concatenate((R,L)), FP).astype(np.uint8)

def dnn_tdes_2key_encrypt(pt, k1, k2, c=0.5):
    a = dnn_des_crypt(pt, k1, c=c)
    b = dnn_des_crypt(a, k2, decrypt=True, c=c)
    return dnn_des_crypt(b, k1, c=c)

def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def run_correctness(rng, outdir, c=0.5, count=10):
    rows = []
    try:
        from Crypto.Cipher import DES3
        crypto_available = True
    except ImportError:
        crypto_available = False
        print('WARNING: pycryptodome not installed; external DES3 comparison skipped')
    for t in range(1, count+1):
        ext1, ext2 = random_external7(rng), random_external7(rng)
        while ext2 == ext1:
            ext2 = random_external7(rng)
        k1, k2 = external7_to_key64(ext1), external7_to_key64(ext2)
        pt = rng.integers(0, 256, 8, dtype=np.uint8).tobytes()
        bits = bytes_to_bits(pt)
        start = time.perf_counter()
        ct_boolean = tdes_2key_encrypt_bits(bits, k1, k2)
        ct_dnn = dnn_tdes_2key_encrypt(bits, k1, k2, c)
        match = bool(np.array_equal(ct_boolean, ct_dnn))
        ct_ref = ''
        ref_match = ''
        if crypto_available:
            from Crypto.Cipher import DES3
            try:
                key_bytes = bits_to_bytes(k1) + bits_to_bytes(k2)
                ct_ref = DES3.new(key_bytes, DES3.MODE_ECB).encrypt(pt).hex().upper()
                ref_match = (ct_ref == bits_to_hex(ct_boolean))
            except ValueError:
                # Rare degenerate effective key pair rejected by DES3.
                ref_match = 'SKIPPED_DEGENERATE_KEY'
        row = dict(trial=t, plaintext=pt.hex().upper(),
                   key1_56=ext1.hex().upper(), key2_56=ext2.hex().upper(),
                   boolean_ct=bits_to_hex(ct_boolean), dnn_ct=bits_to_hex(ct_dnn),
                   reference_ct=ct_ref, boolean_dnn_match=match,
                   reference_match=ref_match,
                   elapsed_sec=round(time.perf_counter()-start, 6))
        rows.append(row)
        print('Correctness %2d/%d: Boolean/DNN=%s reference=%s' % (t,count,match,ref_match))
    write_csv(outdir/'tdes_correctness_10.csv',rows,list(rows[0]))
    return rows

def run_recovery(rng, outdir, count, trials_per_bit, c, eps, progress_every):
    rows=[]
    start=time.perf_counter()
    for t in range(1,count+1):
        ext1, ext2 = random_external7(rng), random_external7(rng)
        r = recover_one_pair(ext1,ext2,trial_index=t,
                             trials_per_bit=trials_per_bit,c=c,eps=eps)
        r['key1_56'] = ext1.hex().upper()
        r['key2_56'] = ext2.hex().upper()
        rows.append(r)
        if t == 1 or t == count or t % progress_every == 0:
            print('Recovery %4d/%4d: pair_exact=%d/%d last=%.3fs' %
                  (t,count,sum(x['pair_exact'] for x in rows),t,r['elapsed']))
    write_csv(outdir/('tdes_key_recovery_%d.csv'%count),rows,list(rows[0]))
    return rows, time.perf_counter()-start

def main_combined():
    CORRECTNESS_TRIALS = 10
    RECOVERY_BATCHES = (10, 1000)
    RANDOM_SEED = 20261027
    TRIALS_PER_BIT = 5
    C = 0.5
    EPS = 1e-2
    PROGRESS_EVERY = 25
    outdir = Path(__file__).resolve().parent / 'results'
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    summary=[]
    print('=== 2-key TDES Boolean / ReLU-DNN correctness ===')
    correctness = run_correctness(rng,outdir,C,CORRECTNESS_TRIALS)
    ok = sum(r['boolean_dnn_match'] for r in correctness)
    ref_ok = sum(r['reference_match'] is True for r in correctness)
    summary.append(dict(experiment='correctness', trials=CORRECTNESS_TRIALS,
                        successes=ok, success_rate=ok/CORRECTNESS_TRIALS,
                        mean_elapsed_sec=mean([r['elapsed_sec'] for r in correctness]),
                        total_elapsed_sec=sum(r['elapsed_sec'] for r in correctness),
                        reference_successes=ref_ok))
    for n in RECOVERY_BATCHES:
        print('\n=== 2-key TDES synthetic DNN key recovery: %d trials ==='%n)
        rows, total = run_recovery(rng,outdir,n,TRIALS_PER_BIT,C,EPS,PROGRESS_EVERY)
        ok = sum(r['pair_exact'] for r in rows)
        summary.append(dict(experiment='key_recovery',trials=n,successes=ok,
                            success_rate=ok/n,
                            mean_elapsed_sec=mean([r['elapsed'] for r in rows]),
                            total_elapsed_sec=total,reference_successes=''))
        print('Pair recovery: %d/%d (%.2f%%), total %.3fs'%(ok,n,100*ok/n,total))
    write_csv(outdir/'tdes_experiment_summary.csv',summary,list(summary[0]))
    print('\nSummary saved to:',outdir)
    for r in summary:
        print(r['experiment'],r['trials'],'success',r['successes'],'/',r['trials'])

if __name__ == '__main__':
    main_combined()
