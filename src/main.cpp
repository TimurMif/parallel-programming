#include <iostream>
#include <ctime>
#include <fstream>
#include <locale>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <mpi.h>

int MATRIX_SIZE = 200;

struct OperationStats
{
    long long multiplications = 0;
    long long additions = 0;
    long long assignments = 0;

    long long total() const
    {
        return multiplications + additions + assignments;
    }
};

void fill_matrix_with_random(std::vector<int>& matrix, int size)
{
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            matrix[i * size + j] = rand() % 10;
}

void multiply_matrices_mpi(const std::vector<int>& A,
    const std::vector<int>& B,
    std::vector<int>& local_result,
    MPI_Comm comm,
    int matrix_size,
    OperationStats& stats)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    stats = OperationStats();
    stats.multiplications = (long long)matrix_size * matrix_size * matrix_size;
    stats.additions = stats.multiplications;
    stats.assignments = (long long)matrix_size * matrix_size * 2;

    int rows_per_proc = matrix_size / size;
    int extra = matrix_size % size;
    int start_row = rank * rows_per_proc + std::min(rank, extra);
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);

    std::vector<int> local(local_rows * matrix_size, 0);

    for (int i = 0; i < local_rows; ++i)
    {
        int global_i = start_row + i;
        for (int j = 0; j < matrix_size; ++j)
        {
            int sum = 0;
            for (int k = 0; k < matrix_size; ++k)
                sum += A[global_i * matrix_size + k] * B[k * matrix_size + j];
            local[i * matrix_size + j] = sum;
        }
    }

    local_result = std::move(local);
}

void calculate_statistics(const std::vector<double>& times,
    double& mean, double& std_dev,
    double& min_time, double& max_time)
{
    if (times.empty()) return;
    double sum = 0;
    for (double t : times) sum += t;
    mean = sum / times.size();

    min_time = *std::min_element(times.begin(), times.end());
    max_time = *std::max_element(times.begin(), times.end());

    double sq_sum = 0;
    for (double t : times)
        sq_sum += (t - mean) * (t - mean);
    std_dev = std::sqrt(sq_sum / times.size());
}

void write_statistics_to_file(int matrix_size, int num_procs,
    const std::vector<double>& times,
    const OperationStats& stats,
    int experiment_number)
{
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    if (global_rank != 0) return;

    static bool first_write = true;
    std::ofstream file("statistics.csv", std::ios::app);
    if (first_write)
    {
        file << "№,Размер,Процессы,Среднее время(с),Мин время(с),Макс время(с),"
            << "Стд.отклонение,Умножения,Сложения,Присваивания,Всего операций\n";
        first_write = false;
    }

    double mean, std_dev, min_time, max_time;
    calculate_statistics(times, mean, std_dev, min_time, max_time);

    file << experiment_number << ","
        << matrix_size << ","
        << num_procs << ","
        << std::fixed << std::setprecision(6) << mean << ","
        << std::setprecision(6) << min_time << ","
        << std::setprecision(6) << max_time << ","
        << std::setprecision(6) << std_dev << ","
        << stats.multiplications << ","
        << stats.additions << ","
        << stats.assignments << ","
        << stats.total() << "\n";
    file.close();
}

void write_summary_report(const std::vector<int>& matrix_sizes,
    const std::vector<int>& proc_counts,
    const std::vector<std::vector<std::vector<double>>>& all_times,
    int num_procs_for_this_run)
{
    int global_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    if (global_rank != 0) return;

    std::ofstream file("report.txt", std::ios::app);
    if (!file.is_open()) return;

    file << "=========================================\n";
    file << "ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ (MPI)\n";
    file << "Тема: Параллельное умножение матриц с MPI\n";
    file << "=========================================\n\n";

    file << "ХАРАКТЕРИСТИКИ СИСТЕМЫ:\n";
    file << "------------------------\n";
    file << "Количество процессов в этом запуске: " << num_procs_for_this_run << "\n";
    file << "Количество замеров для каждого эксперимента: 3\n\n";

    file << "РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ (для " << num_procs_for_this_run << " процессов):\n";
    file << "------------------------\n\n";

    file << "СРЕДНЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ (секунды):\n";
    file << "----------\n";
    file << "Размер\tВремя (среднее ± стд. откл.)\n";
    file << "------\t---------------------------\n";
    for (size_t i = 0; i < matrix_sizes.size(); ++i)
    {
        double sum = 0;
        for (double t : all_times[i][0]) sum += t;
        double mean = sum / all_times[i][0].size();

        double sq_sum = 0;
        for (double t : all_times[i][0]) sq_sum += (t - mean) * (t - mean);
        double std_dev = std::sqrt(sq_sum / all_times[i][0].size());

        file << matrix_sizes[i] << "\t" << std::fixed << std::setprecision(4) << mean
            << " ± " << std::setprecision(4) << std_dev << "\n";
    }

    file << "\n\nУСКОРЕНИЕ (S = T1/Tn) и ЭФФЕКТИВНОСТЬ (E = S/p * 100%) будут рассчитаны по результатам разных запусков.\n";
    file << "Для этого сравните время при 1 процессе с временем при p процессах из разных запусков.\n";

    file << "\n\n=========================================\n";
    file.close();
}

int main(int argc, char** argv)
{
    setlocale(LC_ALL, "ru_RU.UTF-8");

    MPI_Init(&argc, &argv);
    int global_rank, global_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    if (global_size != 4)
    {
        if (global_rank == 0)
        {
            std::cerr << "Error: program must be launched with 4 processes (mpirun -np 4).\n";
        }
        MPI_Finalize();
        return 1;
    }

    srand(static_cast<unsigned>(time(nullptr)));

    std::vector<int> matrix_sizes = { 200, 400, 800, 1200, 1600, 2000 };
    const int NUM_MEASUREMENTS = 3;

    if (global_rank == 0)
    {
        std::cout << "\n=========================================\n";
        std::cout << "LABORATORY WORK: Parallel matrix multiplication with MPI\n";
        std::cout << "=========================================\n";
        std::cout << "The program will sequentially run experiments for 1, 2, and 4 processes.\n";
        std::cout << "WARNING: This will take a lot of time!\n";
    }

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    std::vector<int> procs_list = { 1, 2, 4 };

    for (int nprocs : procs_list)
    {
        MPI_Group sub_group;
        std::vector<int> ranks(nprocs);
        for (int i = 0; i < nprocs; ++i) ranks[i] = i;
        MPI_Group_incl(world_group, nprocs, ranks.data(), &sub_group);

        MPI_Comm subcomm;
        MPI_Comm_create(MPI_COMM_WORLD, sub_group, &subcomm);

        if (subcomm != MPI_COMM_NULL)
        {
            int local_rank, local_size;
            MPI_Comm_rank(subcomm, &local_rank);
            MPI_Comm_size(subcomm, &local_size);

            std::vector<std::vector<std::vector<double>>> all_times;
            int experiment_counter = 1;

            if (local_rank == 0 && global_rank == 0)
            {
                std::cout << "\n=== Starting experiments with " << nprocs << " processes ===\n";
            }

            for (int size : matrix_sizes)
            {
                MATRIX_SIZE = size;
                if (local_rank == 0 && global_rank == 0)
                {
                    std::cout << "\n[ Testing matrix size " << size << " ]\n";
                    std::cout << std::string(50, '-') << "\n";
                }

                std::vector<std::vector<double>> size_times;

                for (int m = 0; m < NUM_MEASUREMENTS; ++m)
                {
                    std::vector<int> A(size * size);
                    std::vector<int> B(size * size);
                    fill_matrix_with_random(A, size);
                    fill_matrix_with_random(B, size);

                    MPI_Barrier(subcomm);

                    double elapsed = 0.0;
                    if (local_rank == 0 && global_rank == 0)
                    {
                        std::cout << "    Measurement " << (m + 1) << "/" << NUM_MEASUREMENTS << "... ";
                        std::cout.flush();
                    }

                    auto start_time = std::chrono::high_resolution_clock::now();

                    std::vector<int> local_result;
                    OperationStats stats;
                    multiply_matrices_mpi(A, B, local_result, subcomm, size, stats);

                    MPI_Barrier(subcomm);

                    auto end_time = std::chrono::high_resolution_clock::now();
                    if (local_rank == 0 && global_rank == 0)
                    {
                        elapsed = std::chrono::duration<double>(end_time - start_time).count();
                        std::cout << std::fixed << std::setprecision(2) << elapsed << " s\n";
                        size_times.push_back({ elapsed });
                    }
                }

                if (local_rank == 0 && global_rank == 0)
                {
                    std::vector<double> times;
                    for (auto& v : size_times)
                        times.push_back(v[0]);

                    OperationStats stats_dummy;
                    stats_dummy.multiplications = (long long)size * size * size;
                    stats_dummy.additions = stats_dummy.multiplications;
                    stats_dummy.assignments = (long long)size * size * 2;

                    write_statistics_to_file(size, nprocs, times, stats_dummy, experiment_counter++);
                    all_times.push_back({ times });
                }
            }

            if (local_rank == 0 && global_rank == 0)
            {
                write_summary_report(matrix_sizes, { nprocs }, all_times, nprocs);
            }

            MPI_Comm_free(&subcomm);
        }

        MPI_Group_free(&sub_group);
    }

    MPI_Group_free(&world_group);
    MPI_Finalize();

    if (global_rank == 0)
    {
        std::cout << "\n=========================================\n";
        std::cout << "All experiments completed successfully!\n";
        std::cout << "Results saved to:\n";
        std::cout << "  - statistics.csv\n";
        std::cout << "  - report.txt\n";
        std::cout << "=========================================\n";
    }

    return 0;
}
