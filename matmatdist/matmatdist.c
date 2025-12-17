/* Autori: Pasquale Miranda, Riccardo Elena */

#include <stdlib.h>
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
  return (a && b) ? (a / gcd(a, b)) * b : 0;
}

void matmatikj(int lda, int ldb, int ldc,
               double *A, double *B, double *C,
               int N1, int N2, int N3)
{
  int i, j, k;
  double aik;
  const double* Arow;
  const double* Brow;
  double* Crow;

  for (i = 0; i < N1; ++i) {
    Arow = A + i * lda;
    Crow = C + i * ldc;
    for (k = 0; k < N2; ++k) {
      aik = Arow[k];
      Brow = B + k * ldb;
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
  MPI_Datatype Atype, Btype;
  int dims[2], periods[2], coords[2];
  int remain[2];
  int NProw, NPcol, K;
  int N1loc, N2loc, N3loc;
  int Acount, Bcount;
  int i, r, c;
  double *Abuf, *Bbuf, *Aptr, *Bptr;

  MPI_Cart_get(Gridcom, 2, dims, periods, coords);
  NProw = dims[0];
  NPcol = dims[1];
  K = lcm(NProw, NPcol);

  N1loc = N1 / NProw;
  N2loc = N2 / K;
  N3loc = N3 / NPcol;

  Acount = lda * (N1loc - 1) + N2loc;
  Bcount = ldb * (N2loc - 1) + N3loc;

  Abuf = (double *)malloc(Acount * sizeof(double));
  Bbuf = (double *)malloc(Bcount * sizeof(double));

  remain[0] = 0;
  remain[1] = 1;
  MPI_Cart_sub(Gridcom, remain, &Rowcom);

  remain[0] = 1;
  remain[1] = 0;
  MPI_Cart_sub(Gridcom, remain, &Colcom);

  MPI_Type_vector(N1loc, N2loc, lda, MPI_DOUBLE, &Atype);
  MPI_Type_vector(N2loc, N3loc, ldb, MPI_DOUBLE, &Btype);
  MPI_Type_commit(&Atype);
  MPI_Type_commit(&Btype);

  for (i = 0; i < K; ++i) {
    c = i % NPcol;
    r = i % NProw;

    Aptr = (coords[1] == c) ? A + (i / NPcol) * N2loc : Abuf;
    Bptr = (coords[0] == r) ? B + (i / NProw) * Bcount : Bbuf;

    MPI_Bcast(Aptr, 1, Atype, c, Rowcom);
    MPI_Bcast(Bptr, 1, Btype, r, Colcom);

    matmatthread(lda, ldb, ldc,
                 Aptr, Bptr, C,
                 N1loc, N2loc, N3loc,
                 dbA, dbB, dbC,
                 NTROW, NTCOL);
  }

  free(Abuf);
  free(Bbuf);
  MPI_Type_free(&Atype);
  MPI_Type_free(&Btype);
  MPI_Comm_free(&Rowcom);
  MPI_Comm_free(&Colcom);
}
