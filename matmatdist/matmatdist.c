/* Autori: Pasquale Miranda, Riccardo Elena */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

/* ========================== Function declaration ========================== */

static int gcd(int a, int b);

static int lcm(int a, int b);

void matmatikj(int lda, int ldb, int ldc,
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

void matmatdist(MPI_Comm Gridcom,
                int lda, int ldb, int ldc,
                double *A, double *B, double *C,
                int N1, int N2, int N3,
                int dbA, int dbB, int dbC,
                int NTROW, int NTCOL);

/* ========================== Function definition =========================== */

static int gcd(int a, int b)
{
  while (b != 0)
  {
    int temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

static int lcm(int a, int b)
{
  if (a == 0 || b == 0)
    return 0;
  return (a / gcd(a, b)) * b;
}

void matmatikj(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  int aik;
  for (i = 0; i < N1; i++)
  {
    for (k = 0; k < N2; k++)
    {
      aik = A[i * lda + k];
      for (j = 0; j < N3; j++)
      {
        // C[i * ldc + j] = C[i * ldc + j] + A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] += aik * B[k * ldb + j];
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

void matmatdist(MPI_Comm Gridcom,
                int lda, int ldb, int ldc,
                double *A, double *B, double *C,
                int N1, int N2, int N3,
                int dbA, int dbB, int dbC,
                int NTROW, int NTCOL)
{
  MPI_Comm Rowcom, Colcom;
  int ndims = 2, dims[2], periods[2], coords[2];
  int NProws, NPcol;
  int N1local, N2local, N3local;
  int directions[2] = {0, 1};
  int K2;
  int i, k, c, r;
  double *Acol, *Brow;

  MPI_Cart_get(Gridcom, ndims, dims, periods, coords);

  NProws = dims[0];
  NPcol = dims[1];

  {
    int a = NPcol, b = NProws;
    int temp;
    while (b != 0)
    {
      temp = b;
      b = a % b;
      a = temp;
    }
    K2 = (a == 0 || b == 0) ? 0 : (NPcol / a) * NProws;
  }

  N1local = N1 / NProws;
  N2local = N2 / K2;
  N3local = N3 / NPcol;

  Acol = (double *)malloc(N1local * N2local * sizeof(double));
  Brow = (double *)malloc(N2local * N3local * sizeof(double));
  MPI_Cart_sub(Gridcom, directions, &Rowcom);
  directions[0] = 1;
  directions[1] = 0;
  MPI_Cart_sub(Gridcom, directions, &Colcom);

  for (k = 0; k < K2; ++k)
  {
    c = k % NPcol;
    r = k % NProws;

    if (coords[0] == r)
      for (i = 0; i < N1local; i++)
        memcpy(&Acol[i * N2local], &A[i * lda], N2local * sizeof(double));

    if (coords[1] == c)
      for (i = 0; i < N2local; i++)
        memcpy(&Brow[i * N3local], &B[i * ldb], N3local * sizeof(double));

    MPI_Bcast(Acol, N1local * N2local, MPI_DOUBLE, r, Colcom);
    MPI_Bcast(Brow, N2local * N3local, MPI_DOUBLE, c, Rowcom);
    matmatthread(N2local, N3local, ldc,
                 Acol, Brow, C,
                 N1local, N2local, N3local,
                 dbA, dbB, dbC,
                 NTROW, NTCOL);
  }

  free(Acol);
  free(Brow);
  MPI_Comm_free(&Rowcom);
  MPI_Comm_free(&Colcom);
}