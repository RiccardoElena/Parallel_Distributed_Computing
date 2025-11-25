/* Autori: Pasquale Miranda, Riccardo Elena */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ========================== Function definition =========================== */

void matmatijk(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  for (int i = 0; i < N1; i++)
  {
    for (int j = 0; j < N3; j++)
    {
      for (int k = 0; k < N2; k++)
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
  for (int j = 0; j < N3; j++)
  {
    for (int k = 0; k < N2; k++)
    {
      for (int i = 0; i < N1; i++)
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
  for (int j = 0; j < N3; j++)
  {
    for (int i = 0; i < N1; i++)
    {
      for (int k = 0; k < N2; k++)
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
  for (int k = 0; k < N2; k++)
  {
    for (int i = 0; i < N1; i++)
    {
      for (int j = 0; j < N3; j++)
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
  for (int k = 0; k < N2; k++)
  {
    for (int j = 0; j < N3; j++)
    {
      for (int i = 0; i < N1; i++)
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
  for (int i = 0; i < N1; i++)
  {
    for (int k = 0; k < N2; k++)
    {
      for (int j = 0; j < N3; j++)
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

  for (int ii = 0; ii < ii_blocs; ii++)
    for (int jj = 0; jj < jj_blocs; jj++)
      for (int kk = 0; kk < kk_blocs; kk++)
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
