#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>

void print_matrix(const std::vector<std::vector<int>>& matrix, const std::string& name) {
    std::cout << name << ":\n";
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << std::setw(4) << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 4; 
    int m = n / size; 

    std::vector<std::vector<int>> A, B, C;

    if (rank == 0) {
        A.resize(n, std::vector<int>(n));
        B.resize(n, std::vector<int>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }
        C.resize(n, std::vector<int>(n, 0));
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> local_A(m * n);
    std::vector<int> local_C(m * n, 0);
    std::vector<int> B_flat(n * n);

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            MPI_Send(&A[i * m][0], m * n, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        std::copy(B[0].begin(), B[0].end(), B_flat.begin());
    }
    MPI_Bcast(B_flat.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Recv(local_A.data(), m * n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                local_C[i * n + j] += local_A[i * n + k] * B_flat[k * n + j];
            }
        }
    }

    // cбор локальных частей матрицы C на процессе 0
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            MPI_Recv(&C[i * m][0], m * n, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_C.data(), m * n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        print_matrix(A, "Матрица A");
        print_matrix(B, "Матрица B");
        print_matrix(C, "Матрица C (результат)");
    }

    MPI_Finalize();
    return 0;
}
