# DES and 2-Key TDES Master-Key Recovery from Synthetic ReLU Activation Leakage

Research proof-of-concept (PoC) implementations for **DES** and **two-key Triple DES (TDES-EDE)**, comparing Boolean and ReLU-based realizations and evaluating master-key recovery under a **synthetic internal-activation leakage oracle**.

> **Scope:** These experiments assume access to selected internal ReLU/XOR activation values. They do **not** demonstrate black-box attacks on standard DES/TDES implementations or physical side-channel key recovery. The reported results are from the included CSV logs and have not been independently rerun for this release.

## Repository structure

```text
DES-TDES-DNN-Key-Recovery/
├── DES/
│   ├── DES_Boolean_DNN_Correctness_KeyRecovery_Combined.py
│   └── results/
│       ├── des_correctness_10.csv
│       ├── des_key_recovery_10.csv
│       ├── des_key_recovery_1000.csv
│       └── des_experiment_summary.csv
├── TDES/
│   ├── TDES_2Key_Boolean_DNN_Correctness_KeyRecovery_Combined.py
│   └── results/
│       ├── tdes_correctness_10.csv
│       ├── tdes_key_recovery_10.csv
│       ├── tdes_key_recovery_1000.csv
│       └── tdes_experiment_summary.csv
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

## Experimental results

| Experiment | DES | 2-key TDES |
| --- | ---: | ---: |
| Boolean / ReLU correctness | 10/10 | 10/10 |
| Key recovery (10 trials) | 10/10 | 10/10 |
| Key recovery (1,000 trials) | 1,000/1,000 | 1,000/1,000 |

The correctness runs compare the Boolean and ReLU-based outputs; the included scripts also provide reference-library comparisons when available. Recovery success is measured against generated keys under the stated synthetic oracle assumptions.

## Requirements

- Python 3.10 recommended
- NumPy
- PyCryptodome (used for reference comparisons where available)

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Reproduce experiments

From the repository root, run:

```powershell
python DES/DES_Boolean_DNN_Correctness_KeyRecovery_Combined.py --seed 20261027
python TDES/TDES_2Key_Boolean_DNN_Correctness_KeyRecovery_Combined.py
```

**DES:** Default runs include 10 correctness checks, 10 recovery trials, and 1,000 recovery trials. To run smaller batches first:

```powershell
python DES/DES_Boolean_DNN_Correctness_KeyRecovery_Combined.py --correctness 10 --small 10 --large 10 --seed 20261027
```

**TDES:** The script runs 10 correctness trials and recovery batches of 10 and 1,000. Its settings are constants in `main_combined()` (including `RANDOM_SEED = 20261027`). Edit `RECOVERY_BATCHES` there to shorten a local test.

Each script writes CSV files into its own `results/` directory, replacing files with the same names. Preserve the supplied experimental logs before rerunning if you need them for archival purposes. The 1,000-trial runs can take tens of minutes.

## Methodology summary

- **DES:** Compare Boolean, natural ReLU and reference implementations; infer round-key information from synthetic internal activations; reconstruct and verify the master key.
- **2-key TDES:** Study EDE composition `E_K1(D_K2(E_K1(P)))`; recover selected round subkeys using synthetic internal leakage, enumerate missing PC-2 bits, and resolve candidates using intermediate-state oracle information.

The attack model is intentionally stronger than observing only public plaintext/ciphertext pairs.

## Citation

If you use this code, please cite the associated paper. Add its final title, author list, venue or ePrint identifier, and year here when available.

Suggested software citation (update the release date/version before publication):

```bibtex
@misc{DES_TDES_DNN_Key_Recovery_Code,
  title        = {DES and 2-Key TDES Master-Key Recovery from Synthetic ReLU Activation Leakage: Research Code},
  author       = {Kim, Kwangjo},
  howpublished = {GitHub repository},
  url          = {https://github.com/kjkim0410/DES-TDES-DNN-Key-Recovery},
  note         = {Research proof-of-concept software}
}
```

The URL above is a **proposed** repository address and will work only after you create and publish the repository.

## License

MIT License; see [LICENSE](LICENSE).
