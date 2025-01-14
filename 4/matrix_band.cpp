#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <ctime>

void initialize_matrix(std::vector<int>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10;
    }
}

void print_matrix(const std::vector<int>& matrix, int rows, int cols, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(4) << matrix[i * cols + j] << " ";
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

    const int N = 4; 
    const int M = 4; 
    const int K = 4; 

    if (N % size != 0) {
        if (rank == 0) {
            std::cerr << "Ошибка: число строк матрицы A должно быть кратно числу процессов!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    const int local_rows = N / size; 

    std::vector<int> A, B(K * M), C;
    std::vector<int> local_A(local_rows * K);
    std::vector<int> local_C(local_rows * M, 0);

    if (rank == 0) {
        A.resize(N * K);
        C.resize(N * M, 0);
        srand(static_cast<unsigned>(time(nullptr)));
        initialize_matrix(A, N, K);
        initialize_matrix(B, K, M);

        print_matrix(A, N, K, "Матрица A");
        print_matrix(B, K, M, "Матрица B");
    }

    MPI_Scatter(A.data(), local_rows * K, MPI_INT, local_A.data(), local_rows * K, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), K * M, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int k = 0; k < K; ++k) {
                local_C[i * M + j] += local_A[i * K + k] * B[k * M + j];
            }
        }
    }

    MPI_Gather(local_C.data(), local_rows * M, MPI_INT, C.data(), local_rows * M, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_matrix(C, N, M, "Матрица C (результат)");
    }

    MPI_Finalize();
    return 0;
}
