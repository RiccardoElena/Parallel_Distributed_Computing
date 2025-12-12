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
  int tmp;

  while (b) {
    tmp = b;
    b = a % b;
    a = tmp;
  }

  return a;
}

static int lcm(int a, int b)
{
  return (a && b) ? 0 : (a / gcd(a, b)) * b;
}

void matmatikj(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  double aik;
  const double* restrict rA = A;
  const double* restrict rB = B;
  double* restrict rC = C;
  const double* Arow;
  const double* Brow;
  double* Crow;

  for (i = 0; i < N1; ++i) {
    Arow = rA + i * lda;
    Crow = rC + i * ldc;
    for (k = 0; k < N2; ++k) {
      aik = Arow[k];
      Brow = rB + k * ldb;
      for (j = 0; j < N3; ++j) {
        Crow[j] += aik * Brow[j];
      }
    }
  }
}

void matmatblock(int lda, int ldb, int ldc,
                 double *A, double *B, double *C,
                 int N1, int N2, int N3,
                 int dbA, int dbB, int dbC)
{
  int nbi, nbk, nbj;
  int ii, kk, jj;
  int ni, nk, nj;

  nbi = (N1 + dbA - 1) / dbA;
  nbk = (N2 + dbB - 1) / dbB;
  nbj = (N3 + dbC - 1) / dbC;
  
  for (ii = 0; ii < nbi; ++ii) {
    ni = (ii < nbi - 1) ? dbA : N1 - ii * dbA;
    for (kk = 0; kk < nbk; ++kk) {
      nk = (kk < nbk - 1) ? dbB : N2 - kk * dbB;
      for (jj = 0; jj < nbj; ++jj) {
        nj = (jj < nbj - 1) ? dbC : N3 - jj * dbC;
        matmatikj(lda, ldb, ldc,
                  A + ii * dbA * lda + kk * dbB,
                  B + kk * dbB * ldb + jj * dbC,
                  C + ii * dbA * ldc + jj * dbC,
                  ni, nk, nj);
      }
    }
  }
}

void matmatthread(int lda, int ldb, int ldc,
                  double *A, double *B, double *C,
                  int N1, int N2, int N3,
                  int dbA, int dbB, int dbC,
                  int NTROW, int NTCOL)
{
#pragma omp parallel num_threads(NTROW * NTCOL)
  {
    int tid, row_id, col_id;
    int row_start, col_start;
    int local_N1, local_N3;
    
    tid = omp_get_thread_num();
    row_id = tid / NTCOL;
    col_id = tid % NTCOL;

    row_start = row_id * (N1 / NTROW);
    col_start = col_id * (N3 / NTCOL);

    local_N1 = (row_id < NTROW -1) ? N1 / NTROW : N1 - row_start;
    local_N3 = (col_id < NTCOL -1) ? N3 / NTCOL : N3 - col_start;

    if (local_N1 > 0 && local_N3 > 0) {
      matmatblock(lda, ldb, ldc,
                  A + row_start * lda,
                  B + col_start,
                  C + row_start * ldc + col_start,
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
  int dims[2], periods[2], coords[2];
  int remain[2];
  int NProw, NPcol, K;
  int N1loc, N2loc, N3loc;
  int i, k, r, c;
  double *Acol, *Brow;

  MPI_Cart_get(Gridcom, 2, dims, periods, coords);
  NProw = dims[0];
  NPcol = dims[1];
  K = lcm(NProw, NPcol);

  N1loc = N1 / NProw;
  N2loc = N2 / K;
  N3loc = N3 / NPcol;

  Acol = (double*) malloc(N1loc * N2loc * sizeof(double));
  Brow = (double*) malloc(N2loc * N3loc * sizeof(double));

  remain[0] = 0;
  remain[1] = 1;

  MPI_Cart_sub(Gridcom, remain, &Rowcom);

  remain[0] = 1;
  remain[1] = 0;

  MPI_Cart_sub(Gridcom, remain, &Colcom);

  for (k = 0; k < K; ++k) {
    r = k % NProw;
    c = k % NPcol;

    if (coords[0] == r) {
      for (i = 0; i < N1loc; ++i) {
        memcpy(Acol + i * N2loc, A + i * lda, N2loc * sizeof(double));
      }
    }

    if (coords[1] == c) {
      for (i = 0; i < N2loc; ++i) {
        memcpy(Brow + i * N3loc, B + i * ldb, N3loc * sizeof(double));
      }
    }

    MPI_Bcast(Acol, N1loc * N2loc, MPI_DOUBLE, r, Colcom);
    MPI_Bcast(Brow, N2loc * N3loc, MPI_DOUBLE, c, Rowcom);

    matmatthread(N2loc, N3loc, ldc,
                 Acol, Brow, C,
                 N1loc, N2loc, N3loc,
                 dbA, dbB, dbC,
                 NTROW, NTCOL);
  }

  free(Acol);
  free(Brow);
  MPI_Comm_free(&Rowcom);
  MPI_Comm_free(&Colcom);
}
