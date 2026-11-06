# Standalone DES correctness + synthetic ReLU activation leakage key recovery
# Experiments: correctness 10, recovery 10, recovery 1000
# Dependencies: numpy, pycryptodome
# Results saved under ./results/
# natural_des_compare_v2.py
# ------------------------------------------------------------
# Boolean DES
# Natural (ReLU corner-sum) DES
# PyCryptodome DES
# 3-way comparison (standalone version)
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
try:
    from Crypto.Cipher import DES
except ImportError:
    DES = None


# ============================================================
# Utilities
# ============================================================

def bytes_to_bits_be(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.uint8)

def bits_to_bytes_be(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8)
    packed = np.packbits(bits, bitorder="big")
    return packed.tobytes()

def hex64_to_bits(hex16: str) -> np.ndarray:
    return bytes_to_bits_be(bytes.fromhex(hex16))

def bits_to_hex64(bits64: np.ndarray) -> str:
    return bits_to_bytes_be(bits64).hex()

def permute(bits: np.ndarray, table_1based: list[int]) -> np.ndarray:
    idx = np.array(table_1based) - 1
    return bits[idx]

def left_rotate(bits: np.ndarray, s: int) -> np.ndarray:
    return np.concatenate([bits[s:], bits[:s]])


# ============================================================
# FIPS 46-3 TABLES
# ============================================================


IP = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]

FP = [
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41,  9, 49, 17, 57, 25
]

E = [
    32,  1,  2,  3,  4,  5,
     4,  5,  6,  7,  8,  9,
     8,  9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32,  1
]

P = [
    16,  7, 20, 21,
    29, 12, 28, 17,
     1, 15, 23, 26,
     5, 18, 31, 10,
     2,  8, 24, 14,
    32, 27,  3,  9,
    19, 13, 30,  6,
    22, 11,  4, 25
]

PC1 = [
    57, 49, 41, 33, 25, 17,  9,
     1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27,
    19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
     7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29,
    21, 13,  5, 28, 20, 12,  4
]

PC2 = [
    14, 17, 11, 24,  1,  5,
     3, 28, 15,  6, 21, 10,
    23, 19, 12,  4, 26,  8,
    16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
]

SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

SBOX = [
    # S1
    [
        [14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7],
        [ 0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8],
        [ 4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0],
        [15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13],
    ],
    # S2
    [
        [15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10],
        [ 3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5],
        [ 0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15],
        [13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9],
    ],
    # S3
    [
        [10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8],
        [13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1],
        [13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7],
        [ 1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12],
    ],
    # S4
    [
        [ 7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15],
        [13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9],
        [10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4],
        [ 3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14],
    ],
    # S5
    [
        [ 2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9],
        [14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6],
        [ 4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14],
        [11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3],
    ],
    # S6
    [
        [12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11],
        [10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8],
        [ 9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6],
        [ 4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13],
    ],
    # S7
    [
        [ 4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1],
        [13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6],
        [ 1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2],
        [ 6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12],
    ],
    # S8
    [
        [13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7],
        [ 1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2],
        [ 7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8],
        [ 2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11],
    ],
]


# ============================================================
# Key schedule
# ============================================================

def des_round_keys_48(key64_bits: np.ndarray):
    key56 = permute(key64_bits, PC1)
    C = key56[:28].copy()
    D = key56[28:].copy()

    subkeys = []
    for s in SHIFTS:
        C = left_rotate(C, s)
        D = left_rotate(D, s)
        CD = np.concatenate([C, D])
        subkeys.append(permute(CD, PC2))
    return subkeys


# ============================================================
# Boolean DES
# ============================================================

def xor_bits(a, b):
    return (a ^ b).astype(np.uint8)

def sbox_lookup_6to4(box_idx, in6):
    b = in6.astype(np.uint8)
    row = (int(b[0]) << 1) | int(b[5])
    col = (int(b[1]) << 3) | (int(b[2]) << 2) | (int(b[3]) << 1) | int(b[4])
    val = SBOX[box_idx][row][col]
    return np.array([(val>>3)&1,(val>>2)&1,(val>>1)&1,val&1],dtype=np.uint8)

def feistel_f_boolean(R32, K48):
    ER = permute(R32, E)
    X  = xor_bits(ER, K48)

    out = []
    for i in range(8):
        out.append(sbox_lookup_6to4(i, X[i*6:(i+1)*6]))
    out = np.concatenate(out)
    return permute(out, P)

def boolean_des_encrypt(pt_bits, key_bits):
    subkeys = des_round_keys_48(key_bits)

    state = permute(pt_bits, IP)
    L, R = state[:32], state[32:]

    for r in range(16):
        f = feistel_f_boolean(R, subkeys[r])
        L, R = R, xor_bits(L, f)

    pre = np.concatenate([R, L])
    return permute(pre, FP)


# ============================================================
# Natural DES (ReLU corner-sum)
# ============================================================

def corner_indicator(x, a, c):
    x = x.astype(np.int32)
    a = a.astype(np.int32)

    k = len(x)
    match = np.sum(1 - np.abs(x - a))
    return max((match - k + c) / c, 0.0)

## Bad
# def corner_indicator(x, a, c):
#     k = len(x)
#     match = np.sum(1 - np.abs(x - a))
#     return max((match - k + c)/c, 0.0)

def sum_of_corners(x, active, c):
    s = 0
    for a in active:
        s += corner_indicator(x, a, c)
    return 1 if s >= 0.5 else 0

def nn_xor2(x, y, c):
    x = int(x)
    y = int(y)
    active = np.array([[0,1],[1,0]], dtype=np.uint8)
    return sum_of_corners(np.array([x,y], dtype=np.uint8), active, c)
### Bad ===
# def nn_xor2(x, y, c):
#     active = np.array([[0,1],[1,0]],dtype=np.uint8)
#     return sum_of_corners(np.array([x,y]), active, c)

class NaturalSBox6to4:

    def __init__(self, table, c):
        self.c = c
        self.active = [[] for _ in range(4)]

        for x in range(64):
            b = np.array([(x>>5)&1,(x>>4)&1,(x>>3)&1,
                          (x>>2)&1,(x>>1)&1,x&1],dtype=np.uint8)
            row = (b[0]<<1)|b[5]
            col = (b[1]<<3)|(b[2]<<2)|(b[3]<<1)|b[4]
            val = table[row][col]
            for j in range(4):
                if (val>>(3-j))&1:
                    self.active[j].append(b)

        self.active = [np.array(a,dtype=np.uint8) for a in self.active]

    def eval(self, x):
        out = np.zeros(4,dtype=np.uint8)
        for j in range(4):
            out[j] = sum_of_corners(x,self.active[j],self.c)
        return out

def feistel_f_natural(R32,K48,sboxes,c):

    ER = permute(R32,E)

    X = np.zeros(48,dtype=np.uint8)
    for i in range(48):
        X[i] = nn_xor2(int(ER[i]),int(K48[i]),c)

    out = []
    for i in range(8):
        out.append(sboxes[i].eval(X[i*6:(i+1)*6]))
    out = np.concatenate(out)

    return permute(out,P)

def natural_des_encrypt(pt_bits,key_bits,c=0.5):

    subkeys = des_round_keys_48(key_bits)
    sboxes = [NaturalSBox6to4(SBOX[i],c) for i in range(8)]

    state = permute(pt_bits,IP)
    L,R = state[:32],state[32:]

    for r in range(16):
        f = feistel_f_natural(R,subkeys[r],sboxes,c)

        newR = np.zeros(32,dtype=np.uint8)
        for i in range(32):
            newR[i] = nn_xor2(int(L[i]),int(f[i]),c)

        L,R = R,newR

    pre = np.concatenate([R,L])
    return permute(pre,FP)


# ============================================================
# PyCryptodome DES
# ============================================================

def cryptodome_des_encrypt(pt_hex,key_hex):
    if DES is None:
        raise ImportError("PyCryptodome is required for independent DES correctness verification. Install: python -m pip install pycryptodome")
    cipher = DES.new(bytes.fromhex(key_hex),DES.MODE_ECB)
    return cipher.encrypt(bytes.fromhex(pt_hex)).hex().upper()



# ============================================================
# Basic helpers
# ============================================================

def bits_to_bytes(bits):
    return np.packbits(bits.astype(np.uint8)).tobytes()

def bytes_to_bits(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)




# ============================================================
# 56-bit external key <-> 64-bit DES key with parity
# ============================================================

def insert_odd_parity(bits56):
    out = np.zeros(64, dtype=np.uint8)
    for i in range(8):
        chunk = bits56[i*7:(i+1)*7]
        parity = 1 ^ (int(np.sum(chunk)) % 2)
        out[i*8:(i+1)*8] = np.concatenate([chunk, [parity]])
    return out

def remove_parity(key64):
    out56 = np.zeros(56, dtype=np.uint8)
    for i in range(8):
        out56[i*7:(i+1)*7] = key64[i*8:(i+1)*8][:7]
    return bits_to_bytes(out56)


# ============================================================
# DNN XOR (real valued)
# ============================================================

_ACTIVE_XOR = np.array([[0.,1.],[1.,0.]])

def corner_real(x,a,c):
    match = np.sum(1-np.abs(x-a))
    return max((match-len(x)+c)/c,0)

def xor_real(x,y,c):
    return sum(corner_real(np.array([x,y]),a,c) for a in _ACTIVE_XOR)


# ============================================================
# rk0 synthetic oracle
# ============================================================

class RK0Oracle:
    def __init__(self,key64,mode,c=0.5,eps=1e-2):
        self.key64=key64
        self.mode=mode
        self.c=c
        self.eps=eps
        self.K1=des_round_keys_48(key64)[0]

    def enc(self,b):
        if self.mode=="sym":
            return float(b)
        return self.eps if b==0 else 1-self.eps

    def query(self,pt):
        state=permute(pt,IP)
        R0=state[32:]
        ER=permute(R0,E)
        z=np.zeros(48)
        for i in range(48):
            z[i]=xor_real(self.enc(int(ER[i])),self.enc(int(self.K1[i])),self.c)
        return z


# ============================================================
# Pair generation
# ============================================================

def genpair(i,rng):
    L0=rng.integers(0,2,32,dtype=np.uint8)
    R0=rng.integers(0,2,32,dtype=np.uint8)
    rp=E[i]-1
    ERi=int(R0[rp])
    R0p=R0.copy()
    R0p[rp]^=1
    pt = permute(np.concatenate([L0,R0]),FP)
    ptp= permute(np.concatenate([L0,R0p]),FP)
    return pt,ptp,ERi


# ============================================================
# Recover K1
# ============================================================

def recover_K1(oracle,mode,trials=5):
    rng=np.random.default_rng()
    Khat=np.zeros(48,dtype=np.uint8)

    for i in range(48):
        votes=0
        for _ in range(trials):
            pt,ptp,ERi=genpair(i,rng)
            z=oracle.query(pt)[i]
            zp=oracle.query(ptp)[i]

            if ERi==0:
                a,b=z,zp
            else:
                a,b=zp,z

            bit=0 if a<b else 1
            votes+=bit
        Khat[i]=1 if votes>trials//2 else 0
    return Khat


# ============================================================
# Complete 56-bit via 256 brute force
# ============================================================

def complete_key(K1,key64_true,c, known_pt=None, known_ct=None):
    pt=np.zeros(64,dtype=np.uint8) if known_pt is None else known_pt
    ct=boolean_des_encrypt(pt,key64_true) if known_ct is None else known_ct

    CD1=np.full(56,-1,dtype=np.int16)
    for j,p in enumerate(PC2):
        CD1[p-1]=K1[j]

    C1=CD1[:28]; D1=CD1[28:]
    C0=np.roll(C1,1); D0=np.roll(D1,1)
    CD0=np.concatenate([C0,D0])
    unknown=np.where(CD0<0)[0]

    for mask in range(256):
        CD=CD0.copy()
        for k,pos in enumerate(unknown):
            CD[pos]=(mask>>k)&1
        key64=np.zeros(64,dtype=np.uint8)
        for j,p in enumerate(PC1):
            key64[p-1]=CD[j]
        for b in range(8):
            byte=key64[b*8:(b+1)*8]
            parity=1^(int(np.sum(byte[:7]))%2)
            byte[7]=parity
            key64[b*8:(b+1)*8]=byte
        if np.array_equal(boolean_des_encrypt(pt,key64),ct):
            return key64
    return None




import argparse
import csv
import time
from pathlib import Path


def random_key(rng):
    bits56 = rng.integers(0, 2, 56, dtype=np.uint8)
    return bits56, insert_odd_parity(bits56)


def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def correctness(n, rng, outdir, c):
    rows = []
    s = time.perf_counter()
    for i in range(1, n + 1):
        pt = rng.integers(0, 2, 64, dtype=np.uint8)
        _, key = random_key(rng)
        ct_b = bits_to_hex64(boolean_des_encrypt(pt, key)).upper()
        ct_n = bits_to_hex64(natural_des_encrypt(pt, key, c)).upper()
        ct_p = cryptodome_des_encrypt(bits_to_hex64(pt), bits_to_hex64(key))
        ok = ct_b == ct_n == ct_p
        rows.append(dict(trial=i, plaintext=bits_to_hex64(pt).upper(), key64=bits_to_hex64(key).upper(),
                         boolean_ct=ct_b, dnn_ct=ct_n, pycryptodome_ct=ct_p, match=ok))
        print(f'[Correctness {i:>2}/{n}] Boolean={ct_b} DNN={ct_n} Reference={ct_p} MATCH={ok}', flush=True)
    save_csv(outdir / 'des_correctness_10.csv', rows, list(rows[0]))
    return sum(r['match'] for r in rows), time.perf_counter() - s


def recovery(n, rng, outdir, c, votes):
    rows = []
    start = time.perf_counter()
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        bits56, key = random_key(rng)
        oracle_sym = RK0Oracle(key, 'sym', c)
        oracle_change = RK0Oracle(key, 'change', c)
        true_k1 = des_round_keys_48(key)[0]
        k1_sym = recover_K1(oracle_sym, 'sym', votes)
        k1_change = recover_K1(oracle_change, 'change', votes)
        pt = rng.integers(0, 2, 64, dtype=np.uint8)
        ct = boolean_des_encrypt(pt, key)
        recovered = complete_key(k1_sym, key, c, pt, ct)
        key_match = recovered is not None and np.array_equal(remove_parity(recovered), bits_to_bytes(bits56))
        # Verify a second independent known plaintext/ciphertext pair.
        pt2 = rng.integers(0, 2, 64, dtype=np.uint8)
        verified = bool(key_match and np.array_equal(boolean_des_encrypt(pt2, recovered), boolean_des_encrypt(pt2, key)))
        elapsed = time.perf_counter() - t0
        row = dict(trial=i, true_key56=bits_to_bytes(bits56).hex().upper(),
                   recovered_key56=remove_parity(recovered).hex().upper() if recovered is not None else '',
                   true_k1=bits_to_bytes(true_k1).hex().upper(),
                   sym_k1=bits_to_bytes(k1_sym).hex().upper(),
                   change_k1=bits_to_bytes(k1_change).hex().upper(),
                   sym_k1_match=bool(np.array_equal(k1_sym, true_k1)),
                   change_k1_match=bool(np.array_equal(k1_change, true_k1)),
                   master_key_match=bool(key_match), independent_pt_match=verified,
                   elapsed_seconds=round(elapsed, 6))
        rows.append(row)
        print(f'[Recovery {i:>4}/{n}] key={row["true_key56"]} recovered={row["recovered_key56"] or "NONE"} '
              f'SYM={row["sym_k1_match"]} CHANGE={row["change_k1_match"]} MATCH={verified} time={elapsed:.3f}s', flush=True)
    save_csv(outdir / f'des_key_recovery_{n}.csv', rows, list(rows[0]))
    return sum(r['independent_pt_match'] for r in rows), time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description='DES correctness and synthetic DNN-leakage master-key recovery')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--correctness', type=int, default=10)
    parser.add_argument('--small', type=int, default=10)
    parser.add_argument('--large', type=int, default=1000)
    parser.add_argument('--votes', type=int, default=5)
    parser.add_argument('--c', type=float, default=0.5)
    args = parser.parse_args()
    if min(args.correctness, args.small, args.large, args.votes) < 1 or not (0 < args.c <= 1):
        parser.error('trial counts and votes must be positive; c must be in (0,1]')
    rng = np.random.default_rng(args.seed)
    outdir = Path(__file__).resolve().parent / 'results'
    outdir.mkdir(exist_ok=True)
    summary = []
    for name, n, fn in [('correctness', args.correctness, correctness),
                        ('key_recovery_10', args.small, recovery),
                        ('key_recovery_1000', args.large, recovery)]:
        print(f'\n=== {name}: {n} trials ===', flush=True)
        if fn is correctness:
            success, elapsed = fn(n, rng, outdir, args.c)
        else:
            success, elapsed = fn(n, rng, outdir, args.c, args.votes)
        summary.append(dict(experiment=name, trials=n, success=success,
                            failures=n-success, success_rate=success/n, elapsed_seconds=round(elapsed, 6)))
        print(f'SUMMARY: {success}/{n} ({100*success/n:.2f}%), {elapsed:.3f} seconds', flush=True)
    save_csv(outdir / 'des_experiment_summary.csv', summary, list(summary[0]))
    print(f'\nCSV output directory: {outdir}', flush=True)
    if any(r['failures'] for r in summary):
        raise SystemExit('One or more experimental checks failed; inspect CSV results.')


if __name__ == '__main__':
    main()
