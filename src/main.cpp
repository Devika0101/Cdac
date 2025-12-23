#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <omp.h>

using namespace std;

const int NUM_SIMULATIONS = 200000;
const double CONFIDENCE_LEVEL = 0.95;

/* ---------- Monte Carlo Function ---------- */
void monteCarlo(bool parallel, int threads,
                double &VaR, double &ES, double &time_taken)
{
    double S0 = 100.0, mu = 0.05, sigma = 0.2, T = 1.0, portfolio = 100.0;
    vector<double> losses(NUM_SIMULATIONS);

    double start = omp_get_wtime();

    if (!parallel)  // SERIAL
    {
        mt19937 gen(42);
        normal_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < NUM_SIMULATIONS; i++)
        {
            double Z = dist(gen);
            double ST = S0 * exp((mu - 0.5 * sigma * sigma) * T +
                                 sigma * sqrt(T) * Z);
            double final_value = portfolio * (ST / S0);
            losses[i] = portfolio - final_value;
        }
    }
    else  // OPENMP PARALLEL
    {
        omp_set_num_threads(threads);

        #pragma omp parallel
        {
            mt19937 gen(42 + omp_get_thread_num());
            normal_distribution<double> dist(0.0, 1.0);

            #pragma omp for
            for (int i = 0; i < NUM_SIMULATIONS; i++)
            {
                double Z = dist(gen);
                double ST = S0 * exp((mu - 0.5 * sigma * sigma) * T +
                                     sigma * sqrt(T) * Z);
                double final_value = portfolio * (ST / S0);
                losses[i] = portfolio - final_value;
            }
        }
    }

    sort(losses.begin(), losses.end());

    int var_index = static_cast<int>((1.0 - CONFIDENCE_LEVEL) * NUM_SIMULATIONS);
    VaR = losses[var_index];

    ES = 0.0;
    for (int i = 0; i <= var_index; i++)
        ES += losses[i];
    ES /= (var_index + 1);

    time_taken = omp_get_wtime() - start;
}

/* ---------- MAIN ---------- */
int main()
{
    double serial_VaR, serial_ES, serial_time;
    monteCarlo(false, 1, serial_VaR, serial_ES, serial_time);

    cout << "\n===== SERIAL RESULTS =====\n";
    cout << "VaR: " << serial_VaR << endl;
    cout << "ES : " << serial_ES << endl;
    cout << "Time: " << serial_time << " seconds\n";

    int thread_counts[] = {1, 2, 4};

    for (int t : thread_counts)
    {
        double VaR, ES, time;
        monteCarlo(true, t, VaR, ES, time);

        cout << "\n===== OPENMP (" << t << " THREADS) =====\n";
        cout << "VaR: " << VaR << endl;
        cout << "ES : " << ES << endl;
        cout << "Time: " << time << " seconds\n";
        cout << "Speedup: " << serial_time / time << endl;

        cout << "VaR Difference: " << fabs(serial_VaR - VaR) << endl;
        cout << "ES Difference : " << fabs(serial_ES - ES) << endl;
    }

    return 0;
}
