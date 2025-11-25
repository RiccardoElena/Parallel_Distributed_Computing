/* Autori: Pasquale Miranda, Riccardo Elena */

#include <string.h>
#include <mpi.h>

/* ========================== Functions' Signatures ==========================*/

void laplace(float *A, float *B, float *daprev, float *danext,
             int N, int LD, int Niter);

void laplace_nb(float *A, float *B, float *daprev, float *danext,
                int N, int LD, int Niter);

/* ======================== Functions' Implementation ========================*/

void laplace(float *A, float *B, float *daprev, float *danext,
             int N, int LD, int Niter)
{
  int myid, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int nlocal = N / nproc;
  int i, j, iter;
  int idx_prev, idx_curr, idx_next;

  for (iter = 0; iter < Niter; ++iter)
  {
    if (myid > 0)
    {
      MPI_Sendrecv(&A[0], N, MPI_FLOAT, myid - 1, 0,
                   daprev, N, MPI_FLOAT, myid - 1, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (myid < nproc - 1)
    {
      MPI_Sendrecv(&A[(nlocal - 1) * LD], N, MPI_FLOAT, myid + 1, 1,
                   danext, N, MPI_FLOAT, myid + 1, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Compute upper boundary (if not first process)
    if (myid > 0)
    {
      idx_curr = 0;
      idx_next = LD;

      for (j = 1; j < N - 1; ++j)
      {
        B[idx_curr + j] = 0.25f * (daprev[j] +
                                   A[idx_next + j] +
                                   A[idx_curr + (j - 1)] +
                                   A[idx_curr + (j + 1)]);
      }
    }

    // Compute interior points (all processes)
    for (i = 1; i < nlocal - 1; ++i)
    {
      idx_prev = (i - 1) * LD;
      idx_curr = i * LD;
      idx_next = (i + 1) * LD;

      for (j = 1; j < N - 1; ++j)
      {
        B[idx_curr + j] = 0.25f * (A[idx_prev + j] +
                                   A[idx_next + j] +
                                   A[idx_curr + (j - 1)] +
                                   A[idx_curr + (j + 1)]);
      }
    }

    // Compute lower boundary (if not last process)
    if (myid < nproc - 1)
    {
      idx_prev = (nlocal - 2) * LD;
      idx_curr = (nlocal - 1) * LD;

      for (j = 1; j < N - 1; ++j)
      {
        B[idx_curr + j] = 0.25f * (A[idx_prev + j] +
                                   danext[j] +
                                   A[idx_curr + (j - 1)] +
                                   A[idx_curr + (j + 1)]);
      }
    }

    // Copy upper boundary results (if not first process)
    if (myid > 0)
    {
      memcpy(&A[1], &B[1], (N - 2) * sizeof(float));
    }

    // Copy interior results (all processes)
    for (i = 1; i < nlocal - 1; ++i)
    {
      idx_curr = i * LD;
      memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));
    }

    // Copy lower boundary results (if not last process)
    if (myid < nproc - 1)
    {
      idx_curr = (nlocal - 1) * LD;
      memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));
    }
  }
}

/*
  TTo optimise communication waiting time, we structured our code
  as asynchronously as possible, doing whatever could be done
  as soon as the required data became available.

  For each iteration, 4 communications are required:
    - receive from previous rank (if not first rank)
    - send to previous rank (if not first rank)
    - receive from next rank (if not last rank)
    - send to next rank (if not last rank)

  Excluding edge cases (which are trivially optimised),
  we can model the possible communication completion order scenarios as a tree.

  The constraints are:
    - Each copy must be done after ALL computation in that branch
    - Each "send" must be done after the corresponding "receive"

  Therefore, labeling the communications as:
    - 0: receive from previous rank
    - 1: receive from next rank
    - 2: send to previous rank
    - 3: send to next rank

  The only possible tree, up to isomorphism, that respects the constraints is:

                 [0|1]
            0 /         \ 1
          [1|2]        [0|3]
        1 /   \ 2    3/     \0
      *[2|3]   [1]   [0]   [2|3]*
      2/  \3    |     |    2/  \3
    [3]   [2]  [3]* *[2]  [3]  [2]

  Where each node is labeled with the communication that MUST be waited for,
  and each branch is labeled with the communication that was completed.

  The * nodes are, for each branch, the first point in computation where internal
  points can be safely copied, and where the copy is actually performed.
*/

void laplace_nb(float *A, float *B, float *daprev, float *danext,
                int N, int LD, int Niter)
{
  int myid, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (N == 0 || nproc == 0)
  {
    return;
  }

  int nlocal = N / nproc;
  int i, j, iter;
  int idx_prev, idx_curr, idx_next, idx_curr_new;
  int index;
  MPI_Request req[4];

  for (iter = 0; iter < Niter; ++iter)
  {
    req[0] = req[1] = req[2] = req[3] = MPI_REQUEST_NULL;
    if (myid > 0)
    {
      MPI_Irecv(daprev, N, MPI_FLOAT, myid - 1, 1,
                MPI_COMM_WORLD, &req[0]);
      MPI_Isend(&A[0], N, MPI_FLOAT, myid - 1, 0,
                MPI_COMM_WORLD, &req[2]);
    }

    if (myid < nproc - 1)
    {
      MPI_Irecv(danext, N, MPI_FLOAT, myid + 1, 0,
                MPI_COMM_WORLD, &req[1]);
      MPI_Isend(&A[(nlocal - 1) * LD], N, MPI_FLOAT, myid + 1, 1,
                MPI_COMM_WORLD, &req[3]);
    }

    for (i = 1; i < nlocal - 1; ++i)
    {
      idx_prev = (i - 1) * LD;
      idx_curr = i * LD;
      idx_next = (i + 1) * LD;

      for (j = 1; j < N - 1; ++j)
      {
        B[idx_curr + j] = 0.25f * (A[idx_prev + j] +
                                   A[idx_next + j] +
                                   A[idx_curr + (j - 1)] +
                                   A[idx_curr + (j + 1)]);
      }
    }

    if (nproc == 1)
    {
      for (i = 1; i < nlocal - 1; ++i)
      {
        idx_curr_new = i * LD;
        memcpy(&A[idx_curr_new + 1], &B[idx_curr_new + 1], (N - 2) * sizeof(float));
      }
      continue;
    }

    if (myid == 0)
    {
      MPI_Wait(&req[1], MPI_STATUS_IGNORE);

      idx_prev = (nlocal - 2) * LD;
      idx_curr = (nlocal - 1) * LD;
      for (j = 1; j < N - 1; ++j)
      {
        B[idx_curr + j] = 0.25f * (A[idx_prev + j] + danext[j] +
                                   A[idx_curr + (j - 1)] + A[idx_curr + (j + 1)]);
      }

      for (i = 1; i < nlocal - 1; ++i)
      {
        idx_curr_new = i * LD;
        memcpy(&A[idx_curr_new + 1], &B[idx_curr_new + 1], (N - 2) * sizeof(float));
      }

      MPI_Wait(&req[3], MPI_STATUS_IGNORE);
      memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));
      continue;
    }

    if (myid == nproc - 1)
    {
      MPI_Wait(&req[0], MPI_STATUS_IGNORE);

      idx_curr = 0;
      idx_next = LD;
      for (j = 1; j < N - 1; ++j)
      {
        B[idx_curr + j] = 0.25f * (daprev[j] + A[idx_next + j] +
                                   A[idx_curr + (j - 1)] + A[idx_curr + (j + 1)]);
      }

      for (i = 1; i < nlocal - 1; ++i)
      {
        idx_curr_new = i * LD;
        memcpy(&A[idx_curr_new + 1], &B[idx_curr_new + 1], (N - 2) * sizeof(float));
      }

      MPI_Wait(&req[2], MPI_STATUS_IGNORE);
      memcpy(&A[1], &B[1], (N - 2) * sizeof(float));
      continue;
    }

    MPI_Request recv_reqs[2] = {req[0], req[1]};
    MPI_Waitany(2, recv_reqs, &index, MPI_STATUS_IGNORE);

    if (index == 0)
    {
      idx_curr = 0;
      idx_next = LD;
      for (j = 1; j < N - 1; ++j)
      {
        B[idx_curr + j] = 0.25f * (daprev[j] + A[idx_next + j] +
                                   A[idx_curr + (j - 1)] + A[idx_curr + (j + 1)]);
      }

      MPI_Request mixed_reqs[2] = {req[1], req[2]};
      MPI_Waitany(2, mixed_reqs, &index, MPI_STATUS_IGNORE);

      if (index == 0)
      {
        idx_prev = (nlocal - 2) * LD;
        idx_curr = (nlocal - 1) * LD;
        for (j = 1; j < N - 1; ++j)
        {
          B[idx_curr + j] = 0.25f * (A[idx_prev + j] + danext[j] +
                                     A[idx_curr + (j - 1)] + A[idx_curr + (j + 1)]);
        }

        for (i = 1; i < nlocal - 1; ++i)
        {
          idx_curr_new = i * LD;
          memcpy(&A[idx_curr_new + 1], &B[idx_curr_new + 1], (N - 2) * sizeof(float));
        }

        MPI_Request send_reqs[2] = {req[2], req[3]};
        MPI_Waitany(2, send_reqs, &index, MPI_STATUS_IGNORE);

        if (index == 0)
        {
          memcpy(&A[1], &B[1], (N - 2) * sizeof(float));
          MPI_Wait(&req[3], MPI_STATUS_IGNORE);
          idx_curr = (nlocal - 1) * LD;
          memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));
        }
        else
        {
          idx_curr = (nlocal - 1) * LD;
          memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));
          MPI_Wait(&req[2], MPI_STATUS_IGNORE);
          memcpy(&A[1], &B[1], (N - 2) * sizeof(float));
        }
      }
      else
      {
        memcpy(&A[1], &B[1], (N - 2) * sizeof(float));

        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        idx_prev = (nlocal - 2) * LD;
        idx_curr = (nlocal - 1) * LD;
        for (j = 1; j < N - 1; ++j)
        {
          B[idx_curr + j] = 0.25f * (A[idx_prev + j] + danext[j] +
                                     A[idx_curr + (j - 1)] + A[idx_curr + (j + 1)]);
        }

        for (i = 1; i < nlocal - 1; ++i)
        {
          idx_curr_new = i * LD;
          memcpy(&A[idx_curr_new + 1], &B[idx_curr_new + 1], (N - 2) * sizeof(float));
        }
        MPI_Wait(&req[3], MPI_STATUS_IGNORE);
        memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));
      }
    }
    else
    {
      idx_prev = (nlocal - 2) * LD;
      idx_curr = (nlocal - 1) * LD;
      for (j = 1; j < N - 1; ++j)
      {
        B[idx_curr + j] = 0.25f * (A[idx_prev + j] + danext[j] +
                                   A[idx_curr + (j - 1)] + A[idx_curr + (j + 1)]);
      }

      MPI_Request mixed_reqs[2] = {req[0], req[3]};
      MPI_Waitany(2, mixed_reqs, &index, MPI_STATUS_IGNORE);

      if (index == 0)
      {
        idx_curr = 0;
        idx_next = LD;
        for (j = 1; j < N - 1; ++j)
        {
          B[idx_curr + j] = 0.25f * (daprev[j] + A[idx_next + j] +
                                     A[idx_curr + (j - 1)] + A[idx_curr + (j + 1)]);
        }
        for (i = 1; i < nlocal - 1; ++i)
        {
          idx_curr_new = i * LD;
          memcpy(&A[idx_curr_new + 1], &B[idx_curr_new + 1], (N - 2) * sizeof(float));
        }
        MPI_Request send_reqs[2] = {req[2], req[3]};
        MPI_Waitany(2, send_reqs, &index, MPI_STATUS_IGNORE);

        if (index == 0)
        {
          memcpy(&A[1], &B[1], (N - 2) * sizeof(float));
          MPI_Wait(&req[3], MPI_STATUS_IGNORE);
          idx_curr = (nlocal - 1) * LD;
          memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));
        }
        else
        {
          idx_curr = (nlocal - 1) * LD;
          memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));
          MPI_Wait(&req[2], MPI_STATUS_IGNORE);
          memcpy(&A[1], &B[1], (N - 2) * sizeof(float));
        }
      }
      else
      {
        idx_curr = (nlocal - 1) * LD;
        memcpy(&A[idx_curr + 1], &B[idx_curr + 1], (N - 2) * sizeof(float));

        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        idx_curr = 0;
        idx_next = LD;
        for (j = 1; j < N - 1; ++j)
        {
          B[idx_curr + j] = 0.25f * (daprev[j] + A[idx_next + j] +
                                     A[idx_curr + (j - 1)] + A[idx_curr + (j + 1)]);
        }
        for (i = 1; i < nlocal - 1; ++i)
        {
          idx_curr_new = i * LD;
          memcpy(&A[idx_curr_new + 1], &B[idx_curr_new + 1], (N - 2) * sizeof(float));
        }
        MPI_Wait(&req[2], MPI_STATUS_IGNORE);
        memcpy(&A[1], &B[1], (N - 2) * sizeof(float));
      }
    }
  }
}
