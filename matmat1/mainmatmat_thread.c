#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMAX 2048
#define NSTART 1024
#define NSTEP 1024
#define BASEFILENAME "threads"
#define BASEFILE "threads.csv"

/* ========================== Function declaration ========================== */

void copymats(int lda, int ldb, int ldc,
              double *A_src, double *B_src, double *C_src,
              double *A_dst, double *B_dst, double *C_dst);

void exec_matmatthread(int lda, int ldb, int ldc,
                       double *A, double *B, double *C,
                       int N1, int N2, int N3,
                       int dbA, int dbB, int dbC,
                       int NTROW, int NTCOL);

void exec_matmatblock(int lda, int ldb, int ldc,
                      double *A, double *B, double *C,
                      int N1, int N2, int N3,
                      int dbA, int dbB, int dbC);

/* ========================== Function definition =========================== */

void copymats(int lda, int ldb, int ldc,
              double *A_src, double *B_src, double *C_src,
              double *A_dst, double *B_dst, double *C_dst)
{
  memcpy(A_dst, A_src, lda * lda * sizeof(double));
  memcpy(B_dst, B_src, ldb * ldb * sizeof(double));
  memcpy(C_dst, C_src, ldc * ldc * sizeof(double));
}

void exec_matmatblock(int lda, int ldb, int ldc,
                      double *A, double *B, double *C,
                      int N1, int N2, int N3,
                      int dbA, int dbB, int dbC)
{
  double *A_copy, *B_copy, *C_copy;
  double get_cur_time(), t1, t2;
  void matmatblock(int, int, int,
                   double *, double *, double *,
                   int, int, int,
                   int, int, int);

  A_copy = (double *)malloc(lda * lda * sizeof(double));
  B_copy = (double *)malloc(ldb * ldb * sizeof(double));
  C_copy = (double *)malloc(ldc * ldc * sizeof(double));

  copymats(lda, ldb, ldc, A, B, C, A_copy, B_copy, C_copy);

  t1 = get_cur_time();
  matmatblock(lda, ldb, ldc, A_copy, B_copy, C_copy,
              N1, N2, N3,
              dbA, dbB, dbC);
  t2 = get_cur_time();

  printf("Check Results:\n");
  printf("C[0][0]=%f\n", C_copy[0 * ldc + 0]);
  printf("C[%d][%d]=%f\n", N1 / 2, N3 / 2, C_copy[(N1 / 2) * ldc + (N3 / 2)]);
  printf("C[%d][%d]=%f\n", N1 - 1, N3 - 1, C_copy[(N1 - 1) * ldc + (N3 - 1)]);

  double time = t2 - t1;
  double perf = (2.0 * N1 * N2 * N3) / time;
  FILE *f = fopen(BASEFILE, "a");
  if (f != NULL)
  {
    fprintf(f, ",%f", perf);
    fflush(f);
    fclose(f);
  }
  else
  {
    fprintf(stderr, "Errore apertura file %s\n", BASEFILE);
    exit(1);
  }
  printf("time=%f\n", time);
  printf("preformance:%e\n\n", perf);

  free(A_copy);
  free(B_copy);
  free(C_copy);
}

void exec_matmatthread(int lda, int ldb, int ldc,
                       double *A, double *B, double *C,
                       int N1, int N2, int N3,
                       int dbA, int dbB, int dbC,
                       int NTROW, int NTCOL)
{
  double *A_copy, *B_copy, *C_copy;
  double get_cur_time(), t1, t2;
  void matmatthread(int, int, int,
                    double *, double *, double *,
                    int, int, int,
                    int, int, int,
                    int, int);

  A_copy = (double *)malloc(lda * lda * sizeof(double));
  B_copy = (double *)malloc(ldb * ldb * sizeof(double));
  C_copy = (double *)malloc(ldc * ldc * sizeof(double));

  copymats(lda, ldb, ldc, A, B, C, A_copy, B_copy, C_copy);

  t1 = get_cur_time();
  matmatthread(lda, ldb, ldc, A_copy, B_copy, C_copy,
               N1, N2, N3,
               dbA, dbB, dbC,
               NTROW, NTCOL);
  t2 = get_cur_time();

  printf("Check Results:\n");
  printf("C[0][0]=%f\n", C_copy[0 * ldc + 0]);
  printf("C[%d][%d]=%f\n", N1 / 2, N3 / 2, C_copy[(N1 / 2) * ldc + (N3 / 2)]);
  printf("C[%d][%d]=%f\n", N1 - 1, N3 - 1, C_copy[(N1 - 1) * ldc + (N3 - 1)]);

  double time = t2 - t1;
  double perf = (2.0 * N1 * N2 * N3) / time;
  FILE *f = fopen(BASEFILE, "a");
  if (f != NULL)
  {
    fprintf(f, ",%f", perf);
    fflush(f);
    fclose(f);
  }
  else
  {
    fprintf(stderr, "Errore apertura file %s\n", BASEFILE);
    exit(1);
  }
  printf("time=%f\n", time);
  printf("preformance:%e\n\n", perf);

  free(A_copy);
  free(B_copy);
  free(C_copy);
}

/* =============================== Main Program ============================= */

int main()
{

  double *A, *B, *C;
  double get_cur_time(), t1, t2;

  int lda, ldb, ldc;
  int N1, N2, N3;
  lda = ldb = ldc = 2500;

  A = (double *)malloc(lda * lda * sizeof(double));
  B = (double *)malloc(ldb * ldb * sizeof(double));
  C = (double *)malloc(ldc * ldc * sizeof(double));

  for (int i = 0; i < NMAX; i++)
  {
    for (int j = 0; j < NMAX; j++)
    {
      C[i * ldc + j] = rand();
    }
  }

  for (int i = 0; i < NMAX; i++)
  {
    for (int j = 0; j < NMAX; j++)
    {
      A[i * lda + j] = rand();
    }
  }

  for (int i = 0; i < NMAX; i++)
  {
    for (int j = 0; j < NMAX; j++)
    {
      B[i * ldb + j] = rand();
    }
  }

  FILE *f = fopen(BASEFILE, "w");
  if (f == NULL)
  {
    fprintf(stderr, "Errore apertura file %s\n", BASEFILE);
    exit(1);
  }

  fprintf(f, "N,block,1 thread,2 threads,4 threads,8 threads\n");
  fclose(f);

  for (int i = NSTART; i <= NMAX; i += NSTEP)
  {
    N1 = N2 = N3 = i;
    f = fopen(BASEFILE, "a");
    if (f == NULL)
    {
      fprintf(stderr, "Errore apertura file %s\n", BASEFILE);
      exit(1);
    }
    fprintf(f, "%d", i);
    fclose(f);

    printf("Esecuzione con N=%d\n", i);
    printf("matmatblock\n");
    exec_matmatblock(lda, ldb, ldc, A, B, C, N1, N2, N3, 256, 256, 256);
    printf("matmatthread 1 thread\n");
    exec_matmatthread(lda, ldb, ldc, A, B, C, N1, N2, N3, 256, 256, 256, 1, 1);
    printf("matmatthread 2 threads\n");
    exec_matmatthread(lda, ldb, ldc, A, B, C, N1, N2, N3, 256, 256, 256, 1, 2);
    printf("matmatthread 4 threads\n");
    exec_matmatthread(lda, ldb, ldc, A, B, C, N1, N2, N3, 256, 256, 256, 2, 2);
    printf("matmatthread 8 threads\n");
    exec_matmatthread(lda, ldb, ldc, A, B, C, N1, N2, N3, 256, 256, 256, 2, 4);
    f = fopen(BASEFILE, "a");
    if (f == NULL)
    {
      fprintf(stderr, "Errore apertura file %s\n", BASEFILE);
      exit(1);
    }

    fprintf(f, "\n");
    fclose(f);
  }
  int ret;
  if ((ret = system("python3 plot_base.py " BASEFILE)) == -1)
  {
    perror("system");
    exit(1);
  }
  else if (ret != 0)
  {
    fprintf(stderr, "plot_base.py terminato con codice %d\n", ret);
  }

  if ((ret = system("python3 plot_base.py " BASEFILE " -o plot_" BASEFILENAME ".png")) == -1)
  {
    perror("system");
    exit(1);
  }
  else if (ret != 0)
  {
    fprintf(stderr, "plot_base.py terminato con codice %d\n", ret);
  }

  free(A);
  free(B);
  free(C);

  return 0;
}
