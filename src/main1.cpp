#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define TOTAL_SIMULATIONS 100000
#define CONFIDENCE 0.95

// Generate uniform random number (0,1)
double uniform_random() {
    return rand() / (double)RAND_MAX;
}

// Generate normal random number (mean=0, std=1)
double normal_random() {
    double u1 = uniform_random();
    double u2 = uniform_random();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int compare(const void *a, const void *b) {
    double x = *(double *)a;
    double y = *(double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char *argv[]) {

    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // number of processes

    // Seed random numbers differently for each process
    srand(time(NULL) + rank);

    int local_N = TOTAL_SIMULATIONS / size;
    double *local_losses = (double *)malloc(local_N * sizeof(double));

    // Monte Carlo simulation
    for (int i = 0; i < local_N; i++) {
        double daily_return = 0.01 * normal_random(); // 1% volatility
        double loss = -daily_return;
        local_losses[i] = loss;
    }

    // Gather all losses to rank 0
    double *all_losses = NULL;
    if (rank == 0) {
        all_losses = (double *)malloc(TOTAL_SIMULATIONS * sizeof(double));
    }

    MPI_Gather(local_losses, local_N, MPI_DOUBLE,
               all_losses, local_N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Sort losses
        qsort(all_losses, TOTAL_SIMULATIONS, sizeof(double), compare);

        int var_index = (int)(CONFIDENCE * TOTAL_SIMULATIONS);
        double VaR = all_losses[var_index];

        // Expected Shortfall
        double sum = 0.0;
        int count = 0;
        for (int i = var_index; i < TOTAL_SIMULATIONS; i++) {
            sum += all_losses[i];
            count++;
        }
        double ES = sum / count;

        printf("VaR (95%%): %f\n", VaR);
        printf("Expected Shortfall: %f\n", ES);
    }

    free(local_losses);
    if (rank == 0) free(all_losses);

    MPI_Finalize();
    return 0;
}
