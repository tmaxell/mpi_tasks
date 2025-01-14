#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 12;
    std::vector<int> vector_a, vector_b;
    std::vector<int> local_a, local_b;

    if (rank == 0) {
        vector_a.resize(n);
        vector_b.resize(n);
        std::cout << "Вектор A: ";
        for (int i = 0; i < n; ++i) {
            vector_a[i] = rand() % 10; 
            vector_b[i] = rand() % 10;
            std::cout << vector_a[i] << " ";
        }
        std::cout << "\nВектор B: ";
        for (int i = 0; i < n; ++i) {
            std::cout << vector_b[i] << " ";
        }
        std::cout << std::endl;
    }

    int local_size = n / size + (rank < n % size ? 1 : 0);
    local_a.resize(local_size);
    local_b.resize(local_size);
    std::vector<int> send_counts(size);
    std::vector<int> displs(size);
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            send_counts[i] = n / size + (i < n % size ? 1 : 0);
            displs[i] = offset;
            offset += send_counts[i];
        }
    }

    MPI_Scatterv(vector_a.data(), send_counts.data(), displs.data(), MPI_INT,
                 local_a.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vector_b.data(), send_counts.data(), displs.data(), MPI_INT,
                 local_b.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_dot_product = 0;
    for (int i = 0; i < local_size; ++i) {
        local_dot_product += local_a[i] * local_b[i];
    }

    int global_dot_product = 0;
    MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Скалярное произведение: " << global_dot_product << std::endl;
    }

    MPI_Finalize();
    return 0;
}
