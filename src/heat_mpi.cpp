/*
 *  MPI program that solves the 2D steady-state heat distribution problem
 *  using the Jacobi iterative method.
 *
 *  Compile:
 *  mpic++ -O3 heat_mpi.cpp -o heat_mpi
 *  mpic++ -O3 heat_mpi.cpp -o heat_mpi_live -DLIVE    (for live refresh of color map)
 *
 *  Execute:
 *  mpiexec -n 4 ./heat_mpi [rows] [cols] [max_its] [filename]
 *
 */

#include <mpi.h> /* MPI communication and wall-clock timer */

#include <algorithm> /* For standard utility functions like std::max */
#include <cmath>     /* For mathematical functions like std::fabs */
#include <cstdlib>   /* For general utility functions like exit() */
#include <fstream>   /* For file output streams (std::ofstream) */
#include <iomanip>   /* For stream formatting manipulators (setw, fixed, setprecision) */
#include <iostream>  /* For standard input/output streams (cout, cerr) */
#include <string>    /* For std::string class and string conversion (stoi, to_string) */
#include <vector>    /* Gather counts and offsets */

using namespace std;

const double EPSILON = 0.001; /* Termination condition */
const int VIEWABLE_SIZE = 20; /* Under this grid size, print array to console directly. */

/* Function prototypes with explicit parameters */
void allocate_2d_array(int r, int c, double*** a);
void initialize_array(int m, int n, double** u, int rank, int size);
void print_solution(const string& filename, int m, int n, double** u, int rank, int size);
int find_steady_state(int m, int n, int max_its, double** u, double** w, double& final_diff,
                      const string& filename, int rank, int size, int global_m);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    double elapsed;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double** u;        /* Previous temperatures */
    double** w;        /* New temperatures */
    int its;           /* Iterations to converge */
    double final_diff; /* Temperature difference at the end */

    /* Default grid size and configuration */
    int m = 200;          /* Number of rows */
    int n = 200;          /* Number of cols */
    int max_its = 100000; /* Maximum number of iterations */
    string filename;      /* Filename of output file */

    /* Extract base name from argv[0] (e.g., "./heat_mpi" -> "heat_mpi") */
    string prog_path(argv[0]);
    size_t pos = prog_path.find_last_of('/');  // position of last slash in the path
    string prog_name = (pos != string::npos) ? prog_path.substr(pos + 1) : prog_path;

    /* Handle optional command-line arguments */
    if (argc > 1) {
        m = stoi(argv[1]);
        n = m;  // Default cols to match rows if only one arg is given
    }
    if (argc > 2) {
        n = stoi(argv[2]);
    }
    if (argc > 3) {
        max_its = stoi(argv[3]);
    }

    filename = prog_name + "-" + to_string(m) + "x" + to_string(n) + ".txt";

    if (argc > 4) {
        filename = argv[4];  // Override by custom filename
    }
    if (argc > 5) {
        if (!rank) cerr << "Usage: " << argv[0] << " [rows] [cols] [max_its] [filename]\n";
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (!rank) {
        cout << "MPI processes: " << size << "\n";
        cout << "Problem size: " << m << "x" << n << "\n";
    }

    int ranked_m = (m - 2) / size + (rank < (m - 2) % size) + 2;

    allocate_2d_array(ranked_m, n, &u);
    allocate_2d_array(ranked_m, n, &w);

    initialize_array(ranked_m, n, u, rank, size);
    initialize_array(ranked_m, n, w, rank, size);

    if (max(m, n) <= VIEWABLE_SIZE) {  // For quick correctness checks
        if (!rank) cout << "Before diffusion:\n";
        print_solution("", m, n, u, rank, size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    its = find_steady_state(ranked_m, n, max_its, u, w, final_diff, filename, rank, size, m);
    double end = MPI_Wtime();
    double local_elapsed = end - start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (max(m, n) <= VIEWABLE_SIZE) {  // For quick correctness checks
        if (!rank) cout << "After diffusion:\n";
        print_solution("", m, n, w, rank, size);
    }

    string status = "Converged";
    if (its > max_its) {
        status = "Not converged";
        its = max_its;
    }

    if (!rank) {
        cout << status << " after " << its << " iterations with max diff: " << fixed
             << setprecision(8) << final_diff << ".\n";
        cout << "Elapsed time = " << fixed << setprecision(6) << elapsed << " sec.\n";
    }

    print_solution(filename, m, n, w, rank, size);
    if (!rank) cout << "Output file " << filename << " created.\n";

    /* Clean up allocated memory */
    delete[] u[0];
    delete[] u;
    delete[] w[0];
    delete[] w;

    MPI_Finalize();
    return EXIT_SUCCESS;
}

/* Allocate a two-dimensional array using contiguous memory */
void allocate_2d_array(int r, int c, double*** a) {
    double* storage = new double[r * c];
    *a = new double*[r];
    for (int i = 0; i < r; i++) {
        (*a)[i] = &storage[i * c];
    }
}

/* Here m includes the two ghost rows; only end ranks have physical row boundaries. */
void initialize_array(int m, int n, double** u, int rank, int size) {
    /* Set initial values and boundary conditions */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            u[i][j] = 25.0; /* Room temperature */
        }
        u[i][0] = 1000.0; /* Heat source */
        u[i][n - 1] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        if (!rank) u[0][j] = 0.0;
        if (rank == size - 1) u[m - 1][j] = 0.0;
    }
}

/* Print the solution to standard output or a file */
void print_solution(const string& filename, int m, int n, double** u, int rank, int size) {
    int rows = (m - 2) / size + (rank < (m - 2) % size);
    vector<int> counts(size), offsets(size);
    double** global = nullptr;
    if (!rank) {
        allocate_2d_array(m, n, &global);
        initialize_array(m, n, global, 0, 1);
        int flag = 1;
        for (int r = 0; r < size; ++r) {
            int rank_rows = (m - 2) / size + (r < (m - 2) % size);
            counts[r] = rank_rows * n;
            offsets[r] = flag * n;
            flag += rank_rows;
        }
    }

    MPI_Gatherv(u[1], rows * n, MPI_DOUBLE, ((rank == 0) ? (global[0]) : (nullptr)), counts.data(),
                offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank) return;

    ostream* outfile = &cout;
    ofstream file_stream;

    if (!filename.empty()) {
        file_stream.open(filename);
        if (!file_stream.is_open()) {
            cerr << "Can't open output file.\n";
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        outfile = &file_stream;
    }

    /* Print the solution array with formatting matching %6.2f\t */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            *outfile << setw(6) << fixed << setprecision(2) << global[i][j] << '\t';
        }
        *outfile << '\n';
    }

    delete[] global[0];
    delete[] global;
}

/* Solve a local stripe: m includes both ghost rows; global_m is used for output. */
int find_steady_state(int m, int n, int max_its, double** u, double** w, double& final_diff,
                      const string& filename, int rank, int size, int global_m) {
    double diff;
    int its;
    int above = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int below = rank == size - 1 ? MPI_PROC_NULL : rank + 1;

#ifndef LIVE
    (void)filename;
    (void)global_m;
#endif

    for (its = 1; its <= max_its; its++) {
        MPI_Sendrecv(u[1], n, MPI_DOUBLE, above, 0, u[m - 1], n, MPI_DOUBLE, below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u[m - 2], n, MPI_DOUBLE, below, 1, u[0], n, MPI_DOUBLE, above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        diff = 0.0;

        /* Compute new temperatures into w */
        for (int i = 1; i < m - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                w[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]);
                double d = fabs(w[i][j] - u[i][j]);
                if (d > diff) {
                    diff = d;
                }
            }
        }

        /* Copy w back to u for the next iteration */
        for (int i = 1; i < m - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                u[i][j] = w[i][j];
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        /* Terminate if convergence is reached */
        if (diff <= EPSILON) break;

#ifdef LIVE
        if (its % 1000 == 0) {
            print_solution(filename, global_m, n, w, rank, size);
        }
#endif
    }
    final_diff = diff;
    return its;
}
