#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <limits>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 12;
    std::vector<int> vector;
    std::vector<int> local_vector;

    if (rank == 0) {
        vector.resize(n);
        std::cout << "Вектор: ";
        for (int i = 0; i < n; ++i) {
            vector[i] = rand() % 100; 
            std::cout << vector[i] << " ";
        }
        std::cout << std::endl;
    }

    int local_size = n / size + (rank < n % size ? 1 : 0);
    local_vector.resize(local_size);

    // распределение данных
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

    MPI_Scatterv(vector.data(), send_counts.data(), displs.data(), MPI_INT,
                 local_vector.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // локальный минимум
    int local_min = std::numeric_limits<int>::max();
    for (int value : local_vector) {
        if (value < local_min) {
            local_min = value;
        }
    }

    // тут ищем глобальный минимум
    int global_min;
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Минимальное значение: " << global_min << std::endl;
    }

    MPI_Finalize();
    return 0;
}
