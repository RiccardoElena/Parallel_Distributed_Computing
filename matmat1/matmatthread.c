/* Autori: Pasquale Miranda, Riccardo Elena */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ========================== Function declaration ========================== */

void matmatikj(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3);

void matmatijk(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3);

void matmatjki(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3);

void matmatjik(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3);

void matmatkij(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3);

void matmatkji(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3);

void matmatblock(int lda, int ldb, int ldc,
                 double *A, double *B, double *C,
                 int N1, int N2, int N3,
                 int dbA, int dbB, int dbC);

void matmatthread(int lda, int ldb, int ldc,
                  double *A, double *B, double *C,
                  int N1, int N2, int N3,
                  int dbA, int dbB, int dbC,
                  int NTROW, int NTCOL);

/* ========================== Function definition =========================== */

void matmatijk(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  for (i = 0; i < N1; i++)
  {
    for (j = 0; j < N3; j++)
    {
      for (k = 0; k < N2; k++)
      {
        // C[i * ldc + j] = C[i * ldc + j] + A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

void matmatjki(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  for (j = 0; j < N3; j++)
  {
    for (k = 0; k < N2; k++)
    {
      for (i = 0; i < N1; i++)
      {
        // C[i * ldc + j] = C[i * ldc + j] + A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

void matmatjik(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  for (j = 0; j < N3; j++)
  {
    for (i = 0; i < N1; i++)
    {
      for (k = 0; k < N2; k++)
      {
        // C[i * ldc + j] = C[i * ldc + j] + A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

void matmatkij(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  for (k = 0; k < N2; k++)
  {
    for (i = 0; i < N1; i++)
    {
      for (j = 0; j < N3; j++)
      {
        // C[i * ldc + j] = C[i * ldc + j] + A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

void matmatkji(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  for (k = 0; k < N2; k++)
  {
    for (j = 0; j < N3; j++)
    {
      for (i = 0; i < N1; i++)
      {
        // C[i * ldc + j] = C[i * ldc + j] + A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

void matmatikj(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  for (i = 0; i < N1; i++)
  {
    for (k = 0; k < N2; k++)
    {
      for (j = 0; j < N3; j++)
      {
        // C[i * ldc + j] = C[i * ldc + j] + A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

void matmatblock(int lda, int ldb, int ldc,
                 double *A, double *B, double *C,
                 int N1, int N2, int N3,
                 int dbA, int dbB, int dbC)
{
  int ii_blocs = N1 / dbA + (N1 % dbA != 0);
  int kk_blocs = N2 / dbB + (N2 % dbB != 0);
  int jj_blocs = N3 / dbC + (N3 % dbC != 0);
  int ii, jj, kk;
  for (ii = 0; ii < ii_blocs; ii++)
    for (jj = 0; jj < jj_blocs; jj++)
      for (kk = 0; kk < kk_blocs; kk++)
      {
        matmatikj(lda, ldb, ldc,
                  &A[(ii * dbA) * lda + (kk * dbB)],
                  &B[(kk * dbB) * ldb + (jj * dbC)],
                  &C[(ii * dbA) * ldc + (jj * dbC)],
                  (ii != ii_blocs - 1) ? dbA : (N1 - ii * dbA),
                  (kk != kk_blocs - 1) ? dbB : (N2 - kk * dbB),
                  (jj != jj_blocs - 1) ? dbC : (N3 - jj * dbC));
      }
}

void matmatthread(int lda, int ldb, int ldc,
                  double *A, double *B, double *C,
                  int N1, int N2, int N3,
                  int dbA, int dbB, int dbC,
                  int NTROW, int NTCOL)
{
#pragma omp parallel num_threads(NTROW *NTCOL)
  {
    int thread_id = omp_get_thread_num();
    int row_id = thread_id / NTCOL;
    int col_id = thread_id % NTCOL;

    int rows_per_thread = N1 / NTROW;
    int cols_per_thread = N3 / NTCOL;

    int row_start = row_id * rows_per_thread;
    int row_end = (row_id == NTROW - 1) ? N1 : row_start + rows_per_thread;

    int col_start = col_id * cols_per_thread;
    int col_end = (col_id == NTCOL - 1) ? N3 : col_start + cols_per_thread;

    int local_N1 = row_end - row_start;
    int local_N3 = col_end - col_start;

    if (local_N1 > 0 && local_N3 > 0)
    {
      matmatblock(lda, ldb, ldc,
                  &A[row_start * lda],
                  &B[col_start],
                  &C[row_start * ldc + col_start],
                  local_N1, N2, local_N3,
                  dbA, dbB, dbC);
    }
  }
}
