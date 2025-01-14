#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

void print_matrix(const std::vector<std::vector<int>>& matrix, const std::string& name, int n) {
    std::cout << name << ":\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(4) << matrix[i][j] << " ";
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
    int q = std::sqrt(size); 

    if (q * q != size) {
        if (rank == 0) {
            std::cerr << "Число процессов должно быть квадратом целого числа." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int block_size = n / q;

    std::vector<std::vector<int>> A, B, C;
    if (rank == 0) {
        A.resize(n, std::vector<int>(n));
        B.resize(n, std::vector<int>(n));
        C.resize(n, std::vector<int>(n, 0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }

        print_matrix(A, "Матрица A", n);
        print_matrix(B, "Матрица B", n);
    }
    std::vector<int> local_A(block_size * block_size);
    std::vector<int> local_B(block_size * block_size);
    std::vector<int> local_C(block_size * block_size, 0);

    // создание топологии сетки
    MPI_Comm grid_comm;
    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    // координаты процесса в сетке
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int row_rank, col_rank;

    // подкоммуникаторы для строк и столбцов
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(grid_comm, coords[1], coords[0], &col_comm);

    if (rank == 0) {
        for (int i = 0; i < q; ++i) {
            for (int j = 0; j < q; ++j) {
                int dest_rank;
                int sub_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, sub_coords, &dest_rank);

                if (dest_rank == 0) {
                    for (int ii = 0; ii < block_size; ++ii) {
                        for (int jj = 0; jj < block_size; ++jj) {
                            local_A[ii * block_size + jj] = A[i * block_size + ii][j * block_size + jj];
                            local_B[ii * block_size + jj] = B[i * block_size + ii][j * block_size + jj];
                        }
                    }
                } else {
                    std::vector<int> block_A(block_size * block_size);
                    std::vector<int> block_B(block_size * block_size);

                    for (int ii = 0; ii < block_size; ++ii) {
                        for (int jj = 0; jj < block_size; ++jj) {
                            block_A[ii * block_size + jj] = A[i * block_size + ii][j * block_size + jj];
                            block_B[ii * block_size + jj] = B[i * block_size + ii][j * block_size + jj];
                        }
                    }
                    MPI_Send(block_A.data(), block_size * block_size, MPI_INT, dest_rank, 0, grid_comm);
                    MPI_Send(block_B.data(), block_size * block_size, MPI_INT, dest_rank, 1, grid_comm);
                }
            }
        }
    } else {
        MPI_Recv(local_A.data(), block_size * block_size, MPI_INT, 0, 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Recv(local_B.data(), block_size * block_size, MPI_INT, 0, 1, grid_comm, MPI_STATUS_IGNORE);
    }

    // алгоритм Фокса
    std::vector<int> temp_A(block_size * block_size);
    for (int step = 0; step < q; ++step) {
        int pivot = (coords[0] + step) % q;
        if (coords[1] == pivot) {
            temp_A = local_A;
        }
        MPI_Bcast(temp_A.data(), block_size * block_size, MPI_INT, pivot, row_comm);

        // локальное умножение
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                for (int k = 0; k < block_size; ++k) {
                    local_C[i * block_size + j] += temp_A[i * block_size + k] * local_B[k * block_size + j];
                }
            }
        }

        // циклический сдвиг столбцов
        MPI_Sendrecv_replace(local_B.data(), block_size * block_size, MPI_INT,
                             (coords[0] + 1) % q, 2,
                             (coords[0] - 1 + q) % q, 2,
                             col_comm, MPI_STATUS_IGNORE);
    }

    // cборка результата
    if (rank == 0) {
        for (int i = 0; i < q; ++i) {
            for (int j = 0; j < q; ++j) {
                int source_rank;
                int sub_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, sub_coords, &source_rank);

                if (source_rank == 0) {
                    for (int ii = 0; ii < block_size; ++ii) {
                        for (int jj = 0; jj < block_size; ++jj) {
                            C[i * block_size + ii][j * block_size + jj] = local_C[ii * block_size + jj];
                        }
                    }
                } else {
                    std::vector<int> block_C(block_size * block_size);
                    MPI_Recv(block_C.data(), block_size * block_size, MPI_INT, source_rank, 3, grid_comm, MPI_STATUS_IGNORE);
                    for (int ii = 0; ii < block_size; ++ii) {
                        for (int jj = 0; jj < block_size; ++jj) {
                            C[i * block_size + ii][j * block_size + jj] = block_C[ii * block_size + jj];
                        }
                    }
                }
            }
        }

        print_matrix(C, "Матрица C (результат)", n);
    } else {
        MPI_Send(local_C.data(), block_size * block_size, MPI_INT, 0, 3, grid_comm);
    }

    MPI_Finalize();
    return 0;
}
