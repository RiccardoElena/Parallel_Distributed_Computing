// usare questo programma chiamante per fare test di correttezza per matmatdist
// SOLO CON GLIGLIE DI PROCESSI (NPROW , NPCOL) = (1,1) e (2,2)

#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  int i, j, Nglob, Mglob, Pglob, lda, mcm;
  int dims[2], period[2], coord[2], TROW, TCOL, rank, size;
  int X, Y, Q, R;
  double *A, *B, *C, *D;
  double time1, time2, Ndouble;
  double get_cur_time();
  MPI_Comm GridCom;

  void matmatdist(MPI_Comm, int, int, int, double *, double *, double *, int, int, int, int, int, int, int, int);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //
  // qua viene definita la griglia di processi
  // ATTENZIONE: il prodotto dims[0]*dims[1] deve essere uguale al
  // numero di processi lanciati da mpirun nel file.pbs
  //
  dims[0] = 1;
  dims[1] = 2;
  period[0] = 1;
  period[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &GridCom);

  //
  // allocazione dello spazio per i test
  //
  lda = 6144;
  A = (double *)malloc(sizeof(double) * lda * lda);
  B = (double *)malloc(sizeof(double) * lda * lda);
  C = (double *)malloc(sizeof(double) * lda * lda);
  D = (double *)malloc(sizeof(double) * lda * lda);

  // ==================================================
  // test di correttezza risultati. Verificare solo per griglie di processi (1,1) e (2,2)
  // ==================================================

  Nglob = 2;
  Mglob = 4;
  Pglob = 4;
  TROW = 1;
  TCOL = 1;

  MPI_Cart_coords(GridCom, rank, 2, coord);

  // il calcolo del mcm serve solo per la stampa del risultato del test di correttezza
  X = dims[0];
  Y = dims[1];
  while (Y != 0)
  {
    Q = X / Y;
    R = X - Q * Y;
    X = Y;
    Y = R;
  }
  mcm = dims[0] * dims[1] / X;

  //
  // definizione delle matrici di input
  //
  for (i = 0; i < Nglob / dims[0]; i++)
  {
    for (j = 0; j < Mglob / mcm; j++)
    {
      A[i * lda + j] = coord[0] * dims[1] * Mglob / mcm + coord[1] * Mglob / mcm + i * Mglob + j;
    }
  }
  for (i = 0; i < Mglob / mcm; i++)
  {
    for (j = 0; j < Pglob / dims[1]; j++)
    {
      B[i * lda + j] = 10 + coord[0] * dims[1] * Pglob + coord[1] * Pglob / dims[1] + i * Pglob + j;
    }
  }
  for (i = 0; i < Nglob / dims[0]; i++)
  {
    for (j = 0; j < Pglob / dims[1]; j++)
    {
      C[i * lda + j] = 0.0;
    }
  }

  matmatdist(GridCom, lda, lda, lda, A, B, C, Nglob, Mglob, Pglob, 1, 1, 1, TROW, TCOL);

  //
  // stampa delle matrici A, B e C
  //
  for (i = 0; i < Nglob / dims[0]; i++)
  {
    for (j = 0; j < Mglob / mcm; j++)
    {
      printf("MAT A id %d->  %f \n", rank, A[i * lda + j]);
    }
  }
  printf("------------------\n");

  for (i = 0; i < Mglob / mcm; i++)
  {
    for (j = 0; j < Pglob / dims[1]; j++)
    {
      printf("MAT B id %d->  %f \n", rank, B[i * lda + j]);
    }
  }
  printf("------------------\n");
  for (i = 0; i < Nglob / dims[0]; i++)
  {
    for (j = 0; j < Pglob / dims[1]; j++)
    {
      printf("MAT C id %d->  %f \n", rank, C[i * lda + j]);
    }
  }

  // ==================================================
  // test di efficienza
  // ==================================================

  srand(0);
  for (i = 0; i < lda; i++)
  {
    for (j = 0; j < lda; j++)
    {
      *(A + i * lda + j) = (float)rand() / RAND_MAX;
      *(B + i * lda + j) = (float)rand() / RAND_MAX;
      *(C + i * lda + j) = (float)rand() / RAND_MAX;
      *(D + i * lda + j) = *(C + i * lda + j);
    }
  }

  if (rank == 0)
    printf("               N         time       Gflops\n");
  for (Nglob = 2048; Nglob <= 2048 * 3; Nglob = Nglob + 2048)
  {
    Ndouble = Nglob;

    TROW = 1;
    TCOL = 1; // test con 1 thread per processo
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = get_cur_time();
    matmatdist(GridCom, lda, lda, lda, A, B, C, Nglob, Nglob, Nglob, 256, 256, 256, TROW, TCOL);
    time2 = get_cur_time() - time1;
    printf(" proc = %d:   %4d   %4d   %e  %f \n", rank, Nglob, TROW * TCOL, time2, 2 * Ndouble * Ndouble * Ndouble / time2 / 1.e9);

    TROW = 2;
    TCOL = 1; // test con 2 thread per processo
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = get_cur_time();
    matmatdist(GridCom, lda, lda, lda, A, B, C, Nglob, Nglob, Nglob, 256, 256, 256, TROW, TCOL);
    time2 = get_cur_time() - time1;
    printf(" proc = %d:   %4d   %4d   %e  %f \n", rank, Nglob, TROW * TCOL, time2, 2 * Ndouble * Ndouble * Ndouble / time2 / 1.e9);

    TROW = 2;
    TCOL = 2; // test con 4 thread per processo
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = get_cur_time();
    matmatdist(GridCom, lda, lda, lda, A, B, C, Nglob, Nglob, Nglob, 256, 256, 256, TROW, TCOL);
    time2 = get_cur_time() - time1;
    printf(" proc = %d:   %4d   %4d   %e  %f \n", rank, Nglob, TROW * TCOL, time2, 2 * Ndouble * Ndouble * Ndouble / time2 / 1.e9);

    TROW = 4;
    TCOL = 2; // test con 4 thread per processo
    MPI_Barrier(MPI_COMM_WORLD);
    time1 = get_cur_time();
    matmatdist(GridCom, lda, lda, lda, A, B, C, Nglob, Nglob, Nglob, 256, 256, 256, TROW, TCOL);
    time2 = get_cur_time() - time1;
    printf(" proc = %d:   %4d   %4d   %e  %f \n", rank, Nglob, TROW * TCOL, time2, 2 * Ndouble * Ndouble * Ndouble / time2 / 1.e9);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
