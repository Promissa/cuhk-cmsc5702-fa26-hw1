/*
 *  Sequential program that solves the 2D steady-state heat distribution problem
 *  using the Jacobi iterative method.
 *
 *  Compile:
 *  c++ -O3 heat_seq.cpp -o heat_seq
 *  c++ -O3 heat_seq.cpp -o heat_seq_live -DLIVE    (for live refresh of color map)
 *
 *  Execute:
 *  ./heat_seq [rows] [cols] [max_its] [filename]
 *
 */

#include <algorithm> /* For standard utility functions like std::max */
#include <chrono>    /* For high-resolution timing (steady_clock, duration) */
#include <cmath>     /* For mathematical functions like std::fabs */
#include <cstdlib>   /* For general utility functions like exit() */
#include <fstream>   /* For file output streams (std::ofstream) */
#include <iomanip>   /* For stream formatting manipulators (setw, fixed, setprecision) */
#include <iostream>  /* For standard input/output streams (cout, cerr) */
#include <string>    /* For std::string class and string conversion (stoi, to_string) */

using namespace std;

const double EPSILON = 0.001; /* Termination condition */
const int VIEWABLE_SIZE = 20; /* Under this grid size, print array to console directly. */

/* Function prototypes with explicit parameters */
void allocate_2d_array(int r, int c, double*** a);
void initialize_array(int m, int n, double** u);
void print_solution(const string& filename, int m, int n, double** u);
int find_steady_state(int m, int n, int max_its, double** u, double** w, double& final_diff,
                      const string& filename);

int main(int argc, char* argv[]) {
    double** u;                       /* Previous temperatures */
    double** w;                       /* New temperatures */
    int its;                          /* Iterations to converge */
    double final_diff;                /* Temperature difference at the end */
    chrono::duration<double> elapsed; /* Execution time */

    /* Default grid size and configuration */
    int m = 200;          /* Number of rows */
    int n = 200;          /* Number of cols */
    int max_its = 100000; /* Maximum number of iterations */
    string filename;      /* Filename of output file */

    /* Extract base name from argv[0] (e.g., "./heat_seq" -> "heat_seq") */
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
        cerr << "Usage: " << argv[0] << " [rows] [cols] [max_its] [filename]\n";
        exit(EXIT_FAILURE);
    }

    cout << "Problem size: " << m << "x" << n << "\n";

    allocate_2d_array(m, n, &u);
    allocate_2d_array(m, n, &w);

    initialize_array(m, n, u);
    initialize_array(m, n, w);

    if (max(m, n) <= VIEWABLE_SIZE) {  // For quick correctness checks
        cout << "Before diffusion:\n";
        print_solution("", m, n, u);
    }

    auto start = chrono::steady_clock::now();
    its = find_steady_state(m, n, max_its, u, w, final_diff, filename);
    auto end = chrono::steady_clock::now();

    elapsed = end - start;

    if (max(m, n) <= VIEWABLE_SIZE) {  // For quick correctness checks
        cout << "After diffusion:\n";
        print_solution("", m, n, w);
    }

    string status = "Converged";
    if (its > max_its) {
        status = "Not converged";
        its = max_its;
    }

    cout << status << " after " << its << " iterations with max diff: " << fixed << setprecision(8)
         << final_diff << ".\n";
    cout << "Elapsed time = " << fixed << setprecision(6) << elapsed.count() << " sec.\n";

    print_solution(filename, m, n, w);
    cout << "Output file " << filename << " created.\n";

    /* Clean up allocated memory */
    delete[] u[0];
    delete[] u;
    delete[] w[0];
    delete[] w;

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

/* Initialize the temperature array */
void initialize_array(int m, int n, double** u) {
    /* Set initial values and boundary conditions */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            u[i][j] = 25.0; /* Room temperature */
        }
        u[i][0] = 1000.0; /* Heat source */
        u[i][n - 1] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        u[0][j] = 0.0;
        u[m - 1][j] = 0.0;
    }
}

/* Print the solution to standard output or a file */
void print_solution(const string& filename, int m, int n, double** u) {
    ostream* outfile = &cout;
    ofstream file_stream;

    if (!filename.empty()) {
        file_stream.open(filename);
        if (!file_stream.is_open()) {
            cerr << "Can't open output file.\n";
            exit(EXIT_FAILURE);
        }
        outfile = &file_stream;
    }

    /* Print the solution array with formatting matching %6.2f\t */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            *outfile << setw(6) << fixed << setprecision(2) << u[i][j] << '\t';
        }
        *outfile << '\n';
    }
}

/* Solve the steady state using Jacobi iteration */
int find_steady_state(int m, int n, int max_its, double** u, double** w, double& final_diff,
                      const string& filename) {
    double diff;
    int its;

#ifndef LIVE
    (void)filename;
#endif

    for (its = 1; its <= max_its; its++) {
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

        /* Terminate if convergence is reached */
        if (diff <= EPSILON) break;

#ifdef LIVE
        if (its % 1000 == 0) {
            print_solution(filename, m, n, w);
        }
#endif
    }
    final_diff = diff;
    return its;
}
