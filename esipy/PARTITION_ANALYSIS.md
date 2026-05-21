# ESIpy Partition Analysis (dev-iao Branch)

This document describes the atomic partitioning schemes available in the `dev-iao` branch, organized by **families**. Each family corresponds to a different mathematical approach or reference basis logic.

---

## **1. Family: ORIGINAL**
Standard partitioning methods and basic IAO variants.
*   **Reference Size (Li):** Mostly 2 (MINAO) or 5 (VALENCE).
*   **Members:**
    *   `mulliken`: Full working basis (Non-orthogonal).
    *   `lowdin`: Full working basis (Symmetric orthogonalization).
    *   `meta-lowdin`: Full working basis (Projective orthogonalization).
    *   `nao`: Natural Atomic Orbitals.
    *   `iao`: Regular IAO using **MINAO** (Size 2 for Li).
    *   `iao2`: `cc-pVTZ` truncated to **STO-3G size** (Size 5 for Li).
    *   `iao-autosad`: IAO based on Spherically Averaged Densities (Size 5 for Li).

---

## **2. Family: EFFAO**
Methods based on "Effective Atomic Orbitals" derived from the density matrix, truncated to the full valence layer (STO-3G size).
*   **Reference Size (Li):** 5 (VALENCE).
*   **Members:**
    *   `iao-effao-gross`
    *   `iao-effao-net`
    *   `iao-effao-lowdin`
    *   `iao-effao-meta-lowdin`
    *   `iao-effao-nao`
    *   `iao-effao-symmetric`
    *   `iao-effao-sps`
    *   `iao-effao-spsa`

---

## **3. Family: FPIAO**
"Flat-Polarized IAO" family. Pure projection methods with scaling factors.
*   **Reference Size (Li):** 10 (POLARIZED - includes $3d$).
*   **Members:**
    *   `fpiao(1.0)`, `fpiao(1.25)`, `fpiao(1.5)`, `fpiao(1.75)`, `fpiao(2.0)`

---

## **4. Family: DFPIAO**
"Dual-Flat-Polarized IAO" family. Hybrids of regular `iao` and `fpiao` with varying weights.
*   **Reference Size (Li):** Mixed (2 and 10).
*   **Members:**
    *   `dfpiao(0.5)`, `dfpiao(0.6)`, `dfpiao(0.7)`, `dfpiao(0.8)`, `dfpiao(0.9)`

---

## **5. Family: PEIAO**
"Polarized-Effao-IAO" family. Includes both pure polarized EFFAOs and their hybrids.
*   **Reference Size (Li):** 10 (POLARIZED) or Mixed.
*   **Members:**
    *   `peiao`: Pure polarized EFFAO (Size 10 for Li).
    *   `dpeiao(w)`: Hybrids of `iao` and `peiao` with weights `0.5, 0.6, 0.7, 0.8, 0.9`.

---

## **Input Keywords ($PARTITION)**

You can use the following keywords to select groups of partitions:

| Keyword | Included Families | Total Partitions |
| :--- | :--- | :--- |
| **`$ALLWIP`** | ORIGINAL + EFFAO + FPIAO + DFPIAO + PEIAO | ~30 |
| **`$ALLEFFAO`**| ORIGINAL + EFFAO | 15 |
| **`$ALLFPIAO`**| ORIGINAL + FPIAO | 12 |
| **`$ALLDFPIAO`**| ORIGINAL + DFPIAO | 12 |
| **`$ALLPEIAO`** | ORIGINAL + PEIAO | 13 |

---

## **Summary of Lithium (Li) Orbitals**
*   **MINAO:** $1s, 2s$ (2)
*   **VALENCE:** $1s, 2s, 2p_x, 2p_y, 2p_z$ (5)
*   **POLARIZED:** $1s, 2s, 2p, 3d_{xy}, 3d_{yz}, 3d_{z^2}, 3d_{xz}, 3d_{x^2-y^2}$ (10)
