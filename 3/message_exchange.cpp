#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "Эта программа требует ровно 2 процесса." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    const int iterations = 1000; 
    std::vector<int> message_sizes = {1, 10, 100, 1000, 10000, 100000}; 
    if (rank == 0) {
        std::cout << "Размер сообщения (байты)\tСреднее время передачи (секунды)" << std::endl;
    }

    for (int n : message_sizes) {
        std::vector<char> buffer(n, 0);

        // cинхронизация процессов перед началом измерений
        MPI_Barrier(MPI_COMM_WORLD);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            if (rank == 0) {
                MPI_Send(buffer.data(), n, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer.data(), n, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buffer.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // среднее время
        double average_time = elapsed.count() / (iterations * 2); // считываем отправку и приём

        if (rank == 0) {
            std::cout << n << "\t\t\t" << average_time << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
