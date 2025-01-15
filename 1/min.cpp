#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <limits>
#include <ctime>
#include <fstream>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::ofstream log_file("process_" + std::to_string(rank) + ".log");
    int n = 12;
    if (argc > 1) {
        n = std::atoi(argv[1]);
        if (n <= 0) {
            if (rank == 0) {
                std::cerr << "Некорректное значение длины вектора. Используется значение по умолчанию: 12." << std::endl;
            }
            n = 12;
        }
    }

    log_file << "Процесс " << rank << ": размер вектора n = " << n << std::endl;
    log_file << "Процесс " << rank << ": количество процессов = " << size << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> vector;
    std::vector<int> local_vector;

    if (rank == 0) {
        vector.resize(n);
        srand(static_cast<unsigned>(time(0))); 
        for (int i = 0; i < n; ++i) {
            vector[i] = rand() % 100;
        }
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

    log_file << "Процесс " << rank << ": локальный минимум = " << local_min << std::endl;

    // глобальный минимум
    int global_min;
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        std::cout << "Размер вектора: " << n << std::endl;
        std::cout << "Количество процессов: " << size << std::endl;
        std::cout << "Минимальное значение: " << global_min << std::endl;
        std::cout << "Время выполнения: " << elapsed_time << " мс" << std::endl;
        log_file << "Процесс " << rank << ": глобальный минимум = " << global_min << std::endl;
        log_file << "Процесс " << rank << ": время выполнения = " << elapsed_time << " мс" << std::endl;
    }

    MPI_Finalize();
    log_file.close();
    return 0;
}
