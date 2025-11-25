/* Autori: Pasquale Miranda, Riccardo Elena */

#include <stddef.h>
#include <math.h>
#include <omp.h>

double maxsum(int N, int LD, double* A, int NT)
{
    double global_max = 0.0;

    #pragma omp parallel num_threads(NT)
    {
        double thread_max = 0.0;
	int i, j;

        #pragma omp for schedule(static) nowait
	for (i = 0; i < N; ++i) {
	    double row_sum = 0.0;
	    double* row_ptr = A + ((size_t)i * LD);

	    for (j = 0; j < N; ++j) {
	        row_sum += sqrt(row_ptr[j]);
	    }

	    if (row_sum > thread_max) {
	        thread_max = row_sum;
	    }
	}

        #pragma omp critical
	{
	    if (thread_max > global_max) {
	        global_max = thread_max;
	    }
	}
    }

    return global_max;
}
