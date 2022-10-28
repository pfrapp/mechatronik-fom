// P. Rapp fuer FOM Mechatronik Veranstaltung.
// Disclaimer: Beispielcode zur Uebung zur Veranschaulichung der
// mathematischen Funktionalitaet -- kein Anspruch auf Codequalitaet.

// Notwendige includes
#include <stdio.h>  // for printf
#include <stdlib.h> // for malloc and free
#include <math.h>   // for fabs

// Matrix structure
typedef struct
{
    int rows;
    int cols;

    float *entries;

} matrix_t;

// Forward declarations
void alloc_matrix(matrix_t *m, int rows, int cols);
void free_matrix(matrix_t *m);
void print_matrix(const matrix_t *m);
int get_entry_index(const matrix_t *m, int row, int col);
float get_entry(const matrix_t *m, int row, int col);
void set_entry(matrix_t *m, int row, int col, float value);
int is_square(const matrix_t *m);
int set_identity(matrix_t *m);
int matrix_multiply(const matrix_t *A, const matrix_t *B, matrix_t *C);
int matrix_inverse(matrix_t *A, matrix_t *B);

// ------------- Implementation --------------

void alloc_matrix(matrix_t *m, int rows, int cols)
{
    m->rows = rows;
    m->cols = cols;

    m->entries = (float *)malloc(sizeof(float) * rows * cols);
}

void free_matrix(matrix_t *m)
{
    // Check if already free
    if (m->entries == NULL)
    {
        return;
    }

    m->rows = 0;
    m->cols = 0;

    free(m->entries);
    m->entries = NULL;
}

void print_matrix(const matrix_t *m)
{
    int row, col;

    for (row = 0; row < m->rows; row++)
    {
        printf("[ ");
        for (col = 0; col < m->cols - 1; col++)
        {
            float entry = get_entry(m, row, col);
            printf("%8.3f, ", entry);
        }
        // Last column
        if (m->cols > 0)
        {
            float entry = get_entry(m, row, col);
            printf("%8.3f", entry);
        }
        printf(" ]\n");
    }
}

int get_entry_index(const matrix_t *m, int row, int col)
{
    if ((row < 0) || (row >= m->rows) || (col < 0) || (col >= m->cols))
    {
        return -1;
    }
    return row + m->rows * col;
}

float get_entry(const matrix_t *m, int row, int col)
{
    int idx = get_entry_index(m, row, col);
    if (idx < 0)
    {
        return -1.0f;
    }
    else
    {
        return m->entries[idx];
    }
}

void set_entry(matrix_t *m, int row, int col, float value)
{
    int idx = get_entry_index(m, row, col);
    if (idx < 0)
    {
        return;
    }
    else
    {
        m->entries[idx] = value;
    }
}

int is_square(const matrix_t *m)
{
    return m->rows == m->cols;
}

int set_identity(matrix_t *m)
{
    // Works only for square matrices
    if (!is_square(m))
    {
        return -1;
    }

    int row, col;
    for (row = 0; row < m->rows; row++)
    {
        for (col = 0; col < m->cols; col++)
        {
            set_entry(m, row, col, row == col);
        }
    }
    return 0;
}

int matrix_multiply(const matrix_t *A, const matrix_t *B, matrix_t *C)
{
    int N;
    int row, col;
    int sum_index;
    float entry_sum;

    // Check dimensions
    if (A->cols != B->rows)
    {
        return -1;
    }

    N = A->cols;
    free_matrix(C);
    alloc_matrix(C, A->rows, B->cols);

    for (row = 0; row < A->rows; row++)
    {
        for (col = 0; col < B->cols; col++)
        {
            entry_sum = 0.0f;
            for (sum_index = 0; sum_index < N; sum_index++)
            {
                entry_sum += get_entry(A, row, sum_index) * get_entry(B, sum_index, col);
            }
            set_entry(C, row, col, entry_sum);
        }
    }

    return 0;
}

int matrix_inverse(matrix_t *A, matrix_t *B)
{
    int N;
    int row, col;

    // Index to specify which unknown variable (x) we are currently considering.
    // In other words, it is the current column of A that we consider.
    // (As opposed to the 'col' index which is used when iterating over the entire matrix).
    // Loops from 0 to N-1.
    int unknown_idx;
    float pivot_value;
    float entry;
    float entry2;

    int max_val_row;
    float max_val;
    

    // Check input
    if (!is_square(A))
    {
        return -1;
    }

    N = A->rows;

    free_matrix(B);
    alloc_matrix(B, N, N);
    set_identity(B);

    for (unknown_idx = 0; unknown_idx < N; unknown_idx++)
    {
        // Exchange rows so as to have the largest entry
        max_val_row = unknown_idx;
        max_val = get_entry(A, max_val_row, unknown_idx);
        for (row = unknown_idx+1; row<A->rows; row++) {
            entry = get_entry(A, row, unknown_idx);
            if (fabs(entry) > fabs(max_val)) {
                max_val = entry;
                max_val_row = row;
            }
        }

        // Swap rows if necessary
        if (max_val_row != unknown_idx) {
            printf("Swapping rows %i and %i\n", max_val_row, unknown_idx);
            for (col=0; col<A->cols; col++) {
                // A matrix
                entry = get_entry(A, max_val_row, col);
                entry2 = get_entry(A, unknown_idx, col);
                set_entry(A, max_val_row, col, entry2);
                set_entry(A, unknown_idx, col, entry);

                // B matrix (same size as A, so no separate loop necessary)
                entry = get_entry(B, max_val_row, col);
                entry2 = get_entry(B, unknown_idx, col);
                set_entry(B, max_val_row, col, entry2);
                set_entry(B, unknown_idx, col, entry);
            }
        }
        

        // Set main diagonal entry to 1.0.
        row = unknown_idx;
        pivot_value = get_entry(A, row, unknown_idx);

        if (fabs(pivot_value) < 1.0e-6)
        {
            printf("Error: Matrix is singular\n");
            return -2;
        }

        // Divide the entire row by the pivot_value in order to achieve this
        for (col = 0; col < N; col++)
        {
            entry = get_entry(A, row, col);
            set_entry(A, row, col, entry / pivot_value);
            entry = get_entry(B, row, col);
            set_entry(B, row, col, entry / pivot_value);
        }

        for (row = 0; row < N; row++)
        {
            if (row == unknown_idx)
            {
                continue;
            }
            pivot_value = -1.0f * get_entry(A, row, unknown_idx);

            for (col = 0; col < N; col++)
            {
                // entry at (row, col) is added with entry at (unknown_idx, col), multiplied by the pivot
                entry2 = get_entry(A, unknown_idx, col);
                entry = get_entry(A, row, col);
                set_entry(A, row, col, entry + entry2 * pivot_value);

                entry2 = get_entry(B, unknown_idx, col);
                entry = get_entry(B, row, col);
                set_entry(B, row, col, entry + entry2 * pivot_value);
            }
        }
    }

    return 0;
}

int main()
{
    matrix_t A, B, C, E;
    matrix_t u, v;
    int ret;

    // Allocate memory
    alloc_matrix(&A, 2, 2);
    alloc_matrix(&B, 2, 2);
    alloc_matrix(&C, 2, 2);
    alloc_matrix(&E, 2, 2);

    // Set matrix entries
    set_entry(&A, 0, 0, 1);
    set_entry(&A, 0, 1, 2);
    set_entry(&A, 1, 0, 3);
    set_entry(&A, 1, 1, 4);

    set_entry(&B, 0, 0, -2);
    set_entry(&B, 0, 1, 5);
    set_entry(&B, 1, 0, 0);
    set_entry(&B, 1, 1, 1);

    matrix_multiply(&A, &B, &C);
    set_identity(&E);

    printf("A=\n");
    print_matrix(&A);
    printf("B=\n");
    print_matrix(&B);
    printf("C=A.B=\n");
    print_matrix(&C);
    printf("E=\n");
    print_matrix(&E);

    printf("Computing B=inv(A)\n");
    ret = matrix_inverse(&A, &B);
    if (ret == 0)
    {
        printf("A=\n");
        print_matrix(&A);
        printf("B=\n");
        print_matrix(&B);
    }

    set_entry(&A, 0, 0, 4);
    set_entry(&A, 0, 1, 2);
    set_entry(&A, 1, 0, 2);
    set_entry(&A, 1, 1, 1);
    printf("A=\n");
    print_matrix(&A);

    printf("Computing B=inv(A)\n");
    ret = matrix_inverse(&A, &B);
    if (ret == 0)
    {
        printf("A=\n");
        print_matrix(&A);
        printf("B=\n");
        print_matrix(&B);
    }

    // Free memory
    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);
    free_matrix(&E);
}