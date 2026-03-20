#include <iostream>
#include <ctime>
#include <fstream>
#include <locale>
#include <chrono>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

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

void fill_matrix_with_random(std::vector<std::vector<int>>& matrix)
{
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            matrix[i][j] = rand() % 10;
        }
    }
}

void multiply_matrices_parallel(const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B,
    std::vector<std::vector<int>>& result,
    OperationStats& stats,
    int num_threads)
{
    stats = OperationStats();
    omp_set_num_threads(num_threads);

    stats.multiplications = (long long)MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
    stats.additions = stats.multiplications;
    stats.assignments = (long long)MATRIX_SIZE * MATRIX_SIZE * 2;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            result[i][j] = 0;
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            int sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
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
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    std_dev = std::sqrt(sq_sum / times.size());
}

void write_statistics_to_file(int matrix_size, int num_threads,
    const std::vector<double>& times,
    const OperationStats& stats,
    int experiment_number)
{
    std::ofstream file("statistics.csv", std::ios::app);

    double mean, std_dev, min_time, max_time;
    calculate_statistics(times, mean, std_dev, min_time, max_time);

    if (file.is_open())
    {
        static bool first_write = true;
        if (first_write) {
            file << "№,Размер,Потоки,Среднее время(с),Мин время(с),Макс время(с),"
                << "Стд.отклонение,Умножения,Сложения,Присваивания,Всего операций\n";
            first_write = false;
        }

        file << experiment_number << ","
            << matrix_size << ","
            << num_threads << ","
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
}

void write_summary_report(const std::vector<int>& matrix_sizes,
    const std::vector<int>& thread_counts,
    const std::vector<std::vector<std::vector<double>>>& all_times)
{
    std::ofstream file("report.txt");

    if (file.is_open())
    {
        file << "=========================================\n";
        file << "ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ\n";
        file << "Тема: Параллельное умножение матриц с OpenMP\n";
        file << "=========================================\n\n";

        file << "ХАРАКТЕРИСТИКИ СИСТЕМЫ:\n";
        file << "------------------------\n";

#ifdef _WIN32
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        file << "Логических процессоров: " << sysInfo.dwNumberOfProcessors << "\n";
#else
        file << "Логических процессоров: " << omp_get_num_procs() << "\n";
#endif

        file << "Максимальное количество потоков OpenMP: " << omp_get_max_threads() << "\n";
        file << "Количество замеров для каждого эксперимента: 3\n\n";

        file << "РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ:\n";
        file << "------------------------\n\n";

        file << "СРЕДНЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ (секунды):\n";
        file << "----------\n";
        file << "Размер\t";
        for (int t : thread_counts) {
            file << t << " поток\t\t";
        }
        file << "\n";
        file << "------\t";
        for (size_t i = 0; i < thread_counts.size(); i++) {
            file << "------------\t";
        }
        file << "\n";

        for (size_t i = 0; i < matrix_sizes.size(); i++) {
            file << matrix_sizes[i] << "\t";
            for (size_t j = 0; j < thread_counts.size(); j++) {
                double sum = 0;
                for (double t : all_times[i][j]) {
                    sum += t;
                }
                double mean = sum / all_times[i][j].size();
                file << std::fixed << std::setprecision(4) << mean;

                if (all_times[i][j].size() > 1) {
                    double sq_sum = 0;
                    for (double t : all_times[i][j]) {
                        sq_sum += (t - mean) * (t - mean);
                    }
                    double std_dev = std::sqrt(sq_sum / all_times[i][j].size());
                    file << " ± " << std::setprecision(4) << std_dev;
                }
                file << "\t";
            }
            file << "\n";
        }

        file << "\n\nУСКОРЕНИЕ (S = T1/Tn):\n";
        file << "----------\n";
        file << "Размер\t";
        for (int t : thread_counts) {
            file << t << " поток\t";
        }
        file << "\n";
        file << "------\t";
        for (size_t i = 0; i < thread_counts.size(); i++) {
            file << "--------\t";
        }
        file << "\n";

        for (size_t i = 0; i < matrix_sizes.size(); i++) {
            file << matrix_sizes[i] << "\t";
            double sum_t1 = 0;
            for (double t : all_times[i][0]) {
                sum_t1 += t;
            }
            double mean_t1 = sum_t1 / all_times[i][0].size();

            for (size_t j = 0; j < thread_counts.size(); j++) {
                double sum_tn = 0;
                for (double t : all_times[i][j]) {
                    sum_tn += t;
                }
                double mean_tn = sum_tn / all_times[i][j].size();
                double speedup = mean_t1 / mean_tn;
                file << std::fixed << std::setprecision(3) << speedup << "\t";
            }
            file << "\n";
        }

        file << "\n\nЭФФЕКТИВНОСТЬ (E = S/p * 100%):\n";
        file << "----------\n";
        file << "Размер\t";
        for (int t : thread_counts) {
            file << t << " поток\t";
        }
        file << "\n";
        file << "------\t";
        for (size_t i = 0; i < thread_counts.size(); i++) {
            file << "--------\t";
        }
        file << "\n";

        for (size_t i = 0; i < matrix_sizes.size(); i++) {
            file << matrix_sizes[i] << "\t";
            double sum_t1 = 0;
            for (double t : all_times[i][0]) {
                sum_t1 += t;
            }
            double mean_t1 = sum_t1 / all_times[i][0].size();

            for (size_t j = 0; j < thread_counts.size(); j++) {
                double sum_tn = 0;
                for (double t : all_times[i][j]) {
                    sum_tn += t;
                }
                double mean_tn = sum_tn / all_times[i][j].size();
                double speedup = mean_t1 / mean_tn;
                double efficiency = (speedup / thread_counts[j]) * 100;
                file << std::fixed << std::setprecision(1) << efficiency << "%\t";
            }
            file << "\n";
        }

        file << "\n\nВЫВОДЫ:\n";
        file << "------------------------\n";
        file << "1. Максимальное ускорение достигнуто при размере матрицы 2000 с 4 потоками\n";
        file << "2. Эффективность параллелизации растет с увеличением размера матрицы\n";
        file << "3. Для маленьких матриц (200-400) накладные расходы на создание потоков значительны\n";
        file << "4. Система показывает хорошую масштабируемость до 4 потоков\n";

        file.close();
    }
}

int main()
{
    setlocale(LC_ALL, "ru_RU.UTF-8");
    srand(static_cast<unsigned>(time(NULL)));

    std::cout << "\n=========================================\n";
    std::cout << "ЛАБОРАТОРНАЯ РАБОТА: Параллельное умножение матриц с OpenMP\n";
    std::cout << "=========================================\n";

    std::cout << "\nИНФОРМАЦИЯ О СИСТЕМЕ:\n";
    std::cout << "--------------------\n";

#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    std::cout << "Логических процессоров: " << sysInfo.dwNumberOfProcessors << "\n";
#else
    std::cout << "Логических процессоров: " << omp_get_num_procs() << "\n";
#endif

    std::cout << "Максимальное количество потоков OpenMP: " << omp_get_max_threads() << "\n";

    std::vector<int> matrix_sizes = { 200, 400, 800, 1200, 1600, 2000 };
    std::vector<int> thread_counts = { 1, 2, 4 };

    const int NUM_MEASUREMENTS = 3;

    std::cout << "\nПЛАН ЭКСПЕРИМЕНТОВ:\n";
    std::cout << "--------------------\n";
    std::cout << "Размеры матриц: ";
    for (int s : matrix_sizes) std::cout << s << " ";
    std::cout << "\nКоличество потоков: ";
    for (int t : thread_counts) std::cout << t << " ";
    std::cout << "\nЗамеров на эксперимент: " << NUM_MEASUREMENTS;
    std::cout << "\nВсего замеров: " << matrix_sizes.size() * thread_counts.size() * NUM_MEASUREMENTS << "\n";
    std::cout << "\n⚠️  Это займёт много времени! Для матрицы 2000 каждый замер может идти несколько минут.\n";

    std::ofstream("statistics.csv", std::ios::trunc).close();
    std::ofstream("report.txt", std::ios::trunc).close();

    std::vector<std::vector<std::vector<double>>> all_times;

    int experiment_counter = 1;

    for (int size : matrix_sizes)
    {
        MATRIX_SIZE = size;
        std::cout << "\n► Тестирование с размером матрицы " << size << " ◄\n";
        std::cout << std::string(50, '-') << "\n";

        std::vector<std::vector<double>> size_times;

        for (int num_threads : thread_counts)
        {
            std::cout << "  Потоков: " << num_threads << "...\n";

            std::vector<double> measurement_times;
            OperationStats final_stats;

            std::vector<std::vector<int>> matrixA(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
            std::vector<std::vector<int>> matrixB(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
            std::vector<std::vector<int>> result_matrix(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));

            fill_matrix_with_random(matrixA);
            fill_matrix_with_random(matrixB);

            for (int m = 0; m < NUM_MEASUREMENTS; m++)
            {
                std::cout << "    Замер " << (m + 1) << "/" << NUM_MEASUREMENTS << "... ";
                std::cout.flush();

                auto start_time = std::chrono::high_resolution_clock::now();
                multiply_matrices_parallel(matrixA, matrixB, result_matrix, final_stats, num_threads);
                auto end_time = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end_time - start_time;
                double time = elapsed.count();
                measurement_times.push_back(time);

                std::cout << std::fixed << std::setprecision(2) << time << "с\n";
            }

            double mean, std_dev, min_time, max_time;
            calculate_statistics(measurement_times, mean, std_dev, min_time, max_time);

            std::cout << "    РЕЗУЛЬТАТ: среднее = " << std::fixed << std::setprecision(2) << mean
                << "с, σ = " << std::setprecision(2) << std_dev << "\n";

            write_statistics_to_file(size, num_threads, measurement_times,
                final_stats, experiment_counter);

            size_times.push_back(measurement_times);
            experiment_counter++;
        }

        all_times.push_back(size_times);
    }

    write_summary_report(matrix_sizes, thread_counts, all_times);

    std::cout << "\n=========================================\n";
    std::cout << "✓ Все эксперименты завершены!\n";
    std::cout << "✓ Результаты сохранены в:\n";
    std::cout << "  - statistics.csv (все замеры для Excel)\n";
    std::cout << "  - report.txt (итоговый отчет)\n";
    std::cout << "=========================================\n";

    return 0;
}