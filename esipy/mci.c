/**
 mci.c

 Four functions exported (C API):
  - double compute_mci_sym(const int *ring, int ring_len,
                           const double *aoms_stacked, int naoms, int m);
  - double compute_mci_natorbs_sym(const int *ring, int ring_len,
                                   const double *aoms_stacked, int naoms, int m,
                                   const double *occ);
  - double compute_mci_nosym(...);
  - double compute_mci_natorbs_nosym(...);

 ring: array of length ring_len containing atom indices (0-based) referencing the AOM stack.
 aoms_stacked: stacked matrices in COLUMN-MAJOR (Fortran) order:
   aoms_stacked is length naoms * m * m; matrix i starts at &aoms_stacked[i*m*m].
 occ (for NO functions) is m x m in column-major.
---------------------------------------------------------------------
 Build (recommended):
   gcc -O3 -fopenmp -fPIC -shared -o libmci.so mci.c -lopenblas
 If you don't have OpenBLAS, remove -lopenblas and BLAS calls fall back
 to the internal naive multiply.
---------------------------------------------------------------------
 Notes:
  - Numpy arrays must be passed as Fortran contiguous (order='F') to match column-major expected here.
  - The code fixes the first element of `ring` (same behaviour as your Python code's permutations(...) sliced to factorial(n-1)),
    and generates permutations of the remaining ring elements.
  - For the "nosym" variants we apply the path[1] < path[-1] filter at leaf nodes to remove reversed permutations.
  - For regular Iring we apply the factor 2^(n-1) to the trace (matching your original Python).
*/


#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Try to include cblas header if available */
#ifdef USE_CBLAS
#include <cblas.h>
#endif

/* Helper: call BLAS dgemm if available, else fallback */
static inline void matmul(const double *A, const double *B, double *C, int m) {
#ifdef USE_CBLAS
    /* Column-major (Fortran) multiply: C = A * B  */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, m, m, 1.0, A, m, B, m, 0.0, C, m);
#else
    /* naive column-major */
    int i, j, k;
    for (j = 0; j < m; ++j) {
        for (i = 0; i < m; ++i) {
            double s = 0.0;
            for (k = 0; k < m; ++k) {
                /* A(i,k) at A[k*m + i] because column-major */
                s += A[k*m + i] * B[j*m + k];
            }
            C[j*m + i] = s;
        }
    }
#endif
}

static inline double mat_trace(const double *A, int m) {
    double tr = 0.0;
    for (int i = 0; i < m; ++i) tr += A[i*m + i]; /* column-major: element (i,i) index i*m + i */
    return tr;
}

/* Multiply product by next_mat and store result into out (allocated by caller). */
static inline void multiply_and_advance(const double *product, const double *next_mat, double *out, int m) {
    matmul(product, next_mat, out, m);
}

/* Create identity matrix (column-major) */
static double *create_identity(int m) {
    double *I = (double*)malloc(sizeof(double) * m * m);
    if (!I) return NULL;
    memset(I, 0, sizeof(double) * m * m);
    for (int i = 0; i < m; ++i) I[i*m + i] = 1.0;
    return I;
}

/* ---------- Internal DFS variants (specialized for each API) ---------- */

/* dfs_sym: symmetric regular Iring (multiply trace by iring_factor; no reverse-filter) */
static void dfs_sym(const int *ring, int n, int m, char *used, int *perm, int pos,
                    const double *product, const double *per_atom_mat, double *sum_acc,
                    double iring_factor)
{
    if (pos == n) {
        double tr = mat_trace(product, m);
        *sum_acc += iring_factor * tr;
        return;
    }
    for (int j = 1; j < n; ++j) {
        if (!used[j]) {
            used[j] = 1;
            perm[pos] = ring[j];
            double *new_product = (double*)malloc(sizeof(double)*m*m);
            if (!new_product) { used[j] = 0; continue; }
            const double *mat_for_atom = per_atom_mat + ((size_t)j * (size_t)m * (size_t)m);
            multiply_and_advance(product, mat_for_atom, new_product, m);
            dfs_sym(ring, n, m, used, perm, pos+1, new_product, per_atom_mat, sum_acc, iring_factor);
            free(new_product);
            used[j] = 0;
        }
    }
}

/* dfs_nosym: non-symmetric regular Iring (apply reverse-filter path[1] < path[n-1]) */
static void dfs_nosym(const int *ring, int n, int m, char *used, int *perm, int pos,
                      const double *product, const double *per_atom_mat, double *sum_acc,
                      double iring_factor)
{
    if (pos == n) {
        if (!(perm[1] < perm[n-1])) return;
        double tr = mat_trace(product, m);
        *sum_acc += iring_factor * tr;
        return;
    }
    for (int j = 1; j < n; ++j) {
        if (!used[j]) {
            used[j] = 1;
            perm[pos] = ring[j];
            double *new_product = (double*)malloc(sizeof(double)*m*m);
            if (!new_product) { used[j] = 0; continue; }
            const double *mat_for_atom = per_atom_mat + ((size_t)j * (size_t)m * (size_t)m);
            multiply_and_advance(product, mat_for_atom, new_product, m);
            dfs_nosym(ring, n, m, used, perm, pos+1, new_product, per_atom_mat, sum_acc, iring_factor);
            free(new_product);
            used[j] = 0;
        }
    }
}

/* dfs_natorb_sym: symmetric natural-orbitals variant (per_atom_mat already contains occ*aom) */
static void dfs_natorb_sym(const int *ring, int n, int m, char *used, int *perm, int pos,
                           const double *product, const double *per_atom_mat, double *sum_acc,
                           double iring_factor)
{
    if (pos == n) {
        double tr = mat_trace(product, m);
        *sum_acc += iring_factor * tr;
        return;
    }
    for (int j = 1; j < n; ++j) {
        if (!used[j]) {
            used[j] = 1;
            perm[pos] = ring[j];
            double *new_product = (double*)malloc(sizeof(double)*m*m);
            if (!new_product) { used[j] = 0; continue; }
            const double *mat_for_atom = per_atom_mat + ((size_t)j * (size_t)m * (size_t)m);
            multiply_and_advance(product, mat_for_atom, new_product, m);
            dfs_natorb_sym(ring, n, m, used, perm, pos+1, new_product, per_atom_mat, sum_acc, iring_factor);
            free(new_product);
            used[j] = 0;
        }
    }
}

/* dfs_natorb_nosym: non-symmetric natural-orbitals variant (apply reverse filter) */
static void dfs_natorb_nosym(const int *ring, int n, int m, char *used, int *perm, int pos,
                              const double *product, const double *per_atom_mat, double *sum_acc,
                              double iring_factor)
{
    if (pos == n) {
        if (!(perm[1] < perm[n-1])) return;
        double tr = mat_trace(product, m);
        *sum_acc += iring_factor * tr;
        return;
    }
    for (int j = 1; j < n; ++j) {
        if (!used[j]) {
            used[j] = 1;
            perm[pos] = ring[j];
            double *new_product = (double*)malloc(sizeof(double)*m*m);
            if (!new_product) { used[j] = 0; continue; }
            const double *mat_for_atom = per_atom_mat + ((size_t)j * (size_t)m * (size_t)m);
            multiply_and_advance(product, mat_for_atom, new_product, m);
            dfs_natorb_nosym(ring, n, m, used, perm, pos+1, new_product, per_atom_mat, sum_acc, iring_factor);
            free(new_product);
            used[j] = 0;
        }
    }
}

/* ---------- Top-level API functions ---------- */

/*
 For the regular Iring we require factor = 2^(n-1).
 For NO variants we use factor = 1.0 and precompute per_atom_mat = occ * aom[ring[j]].
 The per_atom_mat buffer will store matrices for j=0..n-1 in the order of the ring array,
 each matrix size m*m, column-major.
*/

/* Helper to create per_atom matrices for the regular (restricted) variant:
   per_atom_mat[j] = aom[ ring[j] ] (direct pointer copy into buffer).
   We copy the matrices into per_atom_mat buffer in the order j=0..n-1 (matching ring positions).
*/
static void fill_per_atom_from_stack(const int *ring, int n, int m, const double *aoms_stacked, double *per_atom_mat) {
    /* per_atom_mat size n * m * m */
    for (int j = 0; j < n; ++j) {
        int atom_idx = ring[j];
        const double *src = aoms_stacked + ((size_t)atom_idx * (size_t)m * (size_t)m);
        memcpy(per_atom_mat + ((size_t)j * (size_t)m * (size_t)m), src, sizeof(double) * m * m);
    }
}

/* Helper to precompute per_atom_mat = occ * aom[ ring[j] ] */
static void fill_per_atom_occ_mul_aom(const int *ring, int n, int m, const double *aoms_stacked, const double *occ, double *per_atom_mat) {
    double *tmp = (double*)malloc(sizeof(double) * m * m);
    for (int j = 0; j < n; ++j) {
        int atom_idx = ring[j];
        const double *aomj = aoms_stacked + ((size_t)atom_idx * (size_t)m * (size_t)m);
        /* tmp = occ * aomj */
        matmul(occ, aomj, tmp, m);
        memcpy(per_atom_mat + ((size_t)j * (size_t)m * (size_t)m), tmp, sizeof(double) * m * m);
    }
    free(tmp);
}

/* compute_mci_sym:
   - regular Iring
   - symmetric partitions -> we sum permutations starting from ring[0] and then multiply by 0.5
*/
double compute_mci_sym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m) {
    if (ring_len < 2) return 0.0;
    int n = ring_len;
    /* allocate per_atom_mat buffer: copy the AOMs for the ring positions in order */
    double *per_atom_mat = (double*)malloc(sizeof(double) * n * m * m);
    if (!per_atom_mat) return 0.0;
    fill_per_atom_from_stack(ring, n, m, aoms_stacked, per_atom_mat);

    /* precompute identity * per_atom_mat[0] -> first_mat */
    double *I = create_identity(m);
    if (!I) { free(per_atom_mat); return 0.0; }

    double *first_prod = (double*)malloc(sizeof(double) * m * m);
    /* product = I * per_atom_mat[0] */
    multiply_and_advance(I, per_atom_mat + 0 * m * m, first_prod, m);

    double result = 0.0;
    double iring_factor = pow(2.0, (double)(n - 1)); /* regular factor */

    /* We parallelize across choices for the second position (j=1..n-1).
       For each choice j we compute product = first_prod * per_atom_mat[j], then DFS for the rest.
    */
    #pragma omp parallel for reduction(+:result) schedule(dynamic)
    for (int j = 1; j < n; ++j) {
        /* thread-local copy of used / perm / product */
        char *used = (char*)calloc(n, sizeof(char));
        int *perm = (int*)malloc(sizeof(int) * n);
        if (!used || !perm) { free(used); free(perm); continue; }

        used[0] = 1;
        used[j] = 1;
        perm[0] = ring[0];
        perm[1] = ring[j];

        /* compute product2 = first_prod * per_atom_mat[j] */
        double *product2 = (double*)malloc(sizeof(double) * m * m);
        if (!product2) { free(used); free(perm); continue; }
        const double *matj = per_atom_mat + ((size_t)j * (size_t)m * (size_t)m);
        multiply_and_advance(first_prod, matj, product2, m);

        double local_sum = 0.0;
        /* dfs on remaining positions */
        dfs_sym(ring, n, m, used, perm, 2, product2, per_atom_mat, &local_sum, iring_factor);

        result += local_sum;

        free(product2);
        free(used);
        free(perm);
    }

    free(first_prod);
    free(I);
    free(per_atom_mat);

    /* The Python version used 0.5 * sum(...) for partitions 'mulliken' or 'non-symmetric' to account for symmetric AOMs.
       We keep the same convention: multiply by 0.5 here. */
    return 0.5 * result;
}

/* compute_mci_natorbs_sym:
   - NO variant: uses occ
   - symmetric: multiply final sum by 0.5
*/
double compute_mci_natorbs_sym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m, const double *occ) {
    if (ring_len < 2) return 0.0;
    int n = ring_len;
    double *per_atom_mat = (double*)malloc(sizeof(double) * n * m * m);
    if (!per_atom_mat) return 0.0;
    /* per_atom_mat[j] = occ * aom[ ring[j] ] */
    fill_per_atom_occ_mul_aom(ring, n, m, aoms_stacked, occ, per_atom_mat);

    double *I = create_identity(m);
    if (!I) { free(per_atom_mat); return 0.0; }

    double *first_prod = (double*)malloc(sizeof(double) * m * m);
    multiply_and_advance(I, per_atom_mat + 0 * m * m, first_prod, m);

    double result = 0.0;
    double iring_factor = 1.0; /* NO variant doesn't use 2^(n-1) in your original function */

    #pragma omp parallel for reduction(+:result) schedule(dynamic)
    for (int j = 1; j < n; ++j) {
        char *used = (char*)calloc(n, sizeof(char));
        int *perm = (int*)malloc(sizeof(int) * n);
        if (!used || !perm) { free(used); free(perm); continue; }
        used[0] = 1; used[j] = 1;
        perm[0] = ring[0]; perm[1] = ring[j];

        double *product2 = (double*)malloc(sizeof(double) * m * m);
        if (!product2) { free(used); free(perm); continue; }
        const double *matj = per_atom_mat + ((size_t)j * (size_t)m * (size_t)m);
        multiply_and_advance(first_prod, matj, product2, m);

        double local_sum = 0.0;
        dfs_natorb_sym(ring, n, m, used, perm, 2, product2, per_atom_mat, &local_sum, iring_factor);
        result += local_sum;

        free(product2);
        free(used);
        free(perm);
    }

    free(first_prod);
    free(I);
    free(per_atom_mat);

    return 0.5 * result;
}

/* compute_mci_nosym:
   - regular Iring
   - non-symmetric partitions -> we remove reversed permutations by checking path[1] < path[-1]
   - we do NOT multiply by 0.5 (unlike the symmetric case)
*/
double compute_mci_nosym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m) {
    if (ring_len < 2) return 0.0;
    int n = ring_len;
    double *per_atom_mat = (double*)malloc(sizeof(double) * n * m * m);
    if (!per_atom_mat) return 0.0;
    fill_per_atom_from_stack(ring, n, m, aoms_stacked, per_atom_mat);

    double *I = create_identity(m);
    if (!I) { free(per_atom_mat); return 0.0; }

    double *first_prod = (double*)malloc(sizeof(double) * m * m);
    multiply_and_advance(I, per_atom_mat + 0 * m * m, first_prod, m);

    double result = 0.0;
    double iring_factor = pow(2.0, (double)(n - 1)); /* regular factor */

    #pragma omp parallel for reduction(+:result) schedule(dynamic)
    for (int j = 1; j < n; ++j) {
        char *used = (char*)calloc(n, sizeof(char));
        int *perm = (int*)malloc(sizeof(int) * n);
        if (!used || !perm) { free(used); free(perm); continue; }
        used[0] = 1; used[j] = 1;
        perm[0] = ring[0]; perm[1] = ring[j];

        double *product2 = (double*)malloc(sizeof(double) * m * m);
        if (!product2) { free(used); free(perm); continue; }
        const double *matj = per_atom_mat + ((size_t)j * (size_t)m * (size_t)m);
        multiply_and_advance(first_prod, matj, product2, m);

        double local_sum = 0.0;
        dfs_nosym(ring, n, m, used, perm, 2, product2, per_atom_mat, &local_sum, iring_factor);
        result += local_sum;

        free(product2);
        free(used);
        free(perm);
    }

    free(first_prod);
    free(I);
    free(per_atom_mat);

    return result;
}

/* compute_mci_natorbs_nosym:
   - NO variant for non-symmetric partitions (apply path[1] < path[-1] filter)
*/
double compute_mci_natorbs_nosym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m, const double *occ) {
    if (ring_len < 2) return 0.0;
    int n = ring_len;
    double *per_atom_mat = (double*)malloc(sizeof(double) * n * m * m);
    if (!per_atom_mat) return 0.0;
    fill_per_atom_occ_mul_aom(ring, n, m, aoms_stacked, occ, per_atom_mat);

    double *I = create_identity(m);
    if (!I) { free(per_atom_mat); return 0.0; }

    double *first_prod = (double*)malloc(sizeof(double) * m * m);
    multiply_and_advance(I, per_atom_mat + 0 * m * m, first_prod, m);

    double result = 0.0;
    double iring_factor = 1.0; /* NO variant */

    #pragma omp parallel for reduction(+:result) schedule(dynamic)
    for (int j = 1; j < n; ++j) {
        char *used = (char*)calloc(n, sizeof(char));
        int *perm = (int*)malloc(sizeof(int) * n);
        if (!used || !perm) { free(used); free(perm); continue; }
        used[0] = 1; used[j] = 1;
        perm[0] = ring[0]; perm[1] = ring[j];

        double *product2 = (double*)malloc(sizeof(double) * m * m);
        if (!product2) { free(used); free(perm); continue; }
        const double *matj = per_atom_mat + ((size_t)j * (size_t)m * (size_t)m);
        multiply_and_advance(first_prod, matj, product2, m);

        double local_sum = 0.0;
        dfs_natorb_nosym(ring, n, m, used, perm, 2, product2, per_atom_mat, &local_sum, iring_factor);
        result += local_sum;

        free(product2);
        free(used);
        free(perm);
    }

    free(first_prod);
    free(I);
    free(per_atom_mat);

    return result;
}

/* To make symbols visible to dynamic loader when built as shared object */
#if defined(__GNUC__)
    /* no-op: functions are global by default */
#endif
