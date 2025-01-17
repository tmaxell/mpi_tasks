#include <mpi.h>
#include <iostream>
#include <vector>
#include <thread> 
#include <chrono> 
#include <cstdlib> 
#include <unistd.h>

// функция эмуляции вычислений
void emulate_computation(int microseconds) {
    usleep(microseconds); // эмуляция вычислений через задержку
}

// функция эмуляции коммуникаций
void emulate_communication(MPI_Comm comm, int message_size, int repetitions) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<int> message(message_size, rank); // сообщение заполняется номером процесса
    std::vector<int> recv_buffer(message_size);

    for (int i = 0; i < repetitions; ++i) {
        if (rank % 2 == 0) {
            // четные процессы отправляют и принимают
            MPI_Send(message.data(), message_size, MPI_INT, (rank + 1) % size, 0, comm);
            MPI_Recv(recv_buffer.data(), message_size, MPI_INT, (rank + size - 1) % size, 0, comm, MPI_STATUS_IGNORE);
        } else {
            // нечетные процессы принимают и отправляют
            MPI_Recv(recv_buffer.data(), message_size, MPI_INT, (rank + size - 1) % size, 0, comm, MPI_STATUS_IGNORE);
            MPI_Send(message.data(), message_size, MPI_INT, (rank + 1) % size, 0, comm);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int computation_time = 1000;  // время вычислений (в микросекундах)
    int message_size = 100;       // размер сообщений (в элементах int)
    int repetitions = 10;         // количество повторений передачи сообщений
    if (argc > 1) computation_time = std::atoi(argv[1]); // объем вычислений
    if (argc > 2) message_size = std::atoi(argv[2]);     // объем коммуникаций
    if (argc > 3) repetitions = std::atoi(argv[3]);      // число повторений
    if (rank == 0) {
        std::cout << "Запуск с параметрами:" << std::endl;
        std::cout << "Объем вычислений: " << computation_time << " мкс" << std::endl;
        std::cout << "Размер сообщений: " << message_size << " элементов" << std::endl;
        std::cout << "Число повторений: " << repetitions << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD); // Синхронизация перед началом
    double start_time = MPI_Wtime();
    emulate_computation(computation_time);
    emulate_communication(MPI_COMM_WORLD, message_size, repetitions);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    std::cout << "Процесс " << rank << ": время выполнения = " << elapsed_time << " секунд" << std::endl;

    MPI_Finalize();
    return 0;
}
