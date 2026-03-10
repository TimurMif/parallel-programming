#include <iostream>
#include <ctime>
#include <fstream>
#include <locale>
#include <chrono>
#include <vector>

const int MATRIX_SIZE = 2000;  

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

void multiply_matrices(const std::vector<std::vector<int>>& A,
                       const std::vector<std::vector<int>>& B,
                       std::vector<std::vector<int>>& result,
                       OperationStats &stats)
{
    stats = OperationStats();

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            result[i][j] = 0;
            stats.assignments++;
        }
    }

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            for (int k = 0; k < MATRIX_SIZE; k++)
            {
                result[i][j] += A[i][k] * B[k][j];
                stats.multiplications++;
                stats.additions++;
            }
        }
    }
}

void write_matrices_to_file(const std::vector<std::vector<int>>& A,
                            const std::vector<std::vector<int>>& B,
                            int experiment_number)
{
    std::ofstream file("input.txt", std::ios::app);

    if (file.is_open())
    {
        file << "Эксперимент #" << experiment_number << std::endl;
        file << "Размер матриц: " << MATRIX_SIZE << std::endl;

        file << "Матрица A:" << std::endl;
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                file << A[i][j] << " ";
            }
            file << std::endl;
        }

        file << "Матрица B:" << std::endl;
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                file << B[i][j] << " ";
            }
            file << std::endl;
        }
        file << "------------------------" << std::endl;
        file.close();
    }
}

void write_result_to_file(const std::vector<std::vector<int>>& result,
                          double elapsed_time,
                          const OperationStats &stats,
                          int experiment_number)
{
    std::ofstream file("result.txt", std::ios::app);

    if (file.is_open())
    {
        file << "Эксперимент #" << experiment_number << std::endl;
        file << "Время: " << elapsed_time << " с" << std::endl;
        file << "Размер матриц: " << MATRIX_SIZE << std::endl;
        file << "Объем: " << stats.total() << std::endl;

        file << "Результирующая матрица:" << std::endl;
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                file << result[i][j] << " ";
            }
            file << std::endl;
        }
        file << "------------------------" << std::endl;
        file.close();
    }
}

void write_data_to_file(int experiment_number, double elapsed_time, const OperationStats &stats)
{
    std::ofstream file("data.txt", std::ios::app);

    if (file.is_open())
    {
        file << "Размер: " << MATRIX_SIZE << std::endl;
        file << "Время: " << elapsed_time << " с" << std::endl;
        file << "Объем: " << stats.total() << std::endl;
        file << "------" << std::endl;
        file.close();
    }
}

int main()
{
    setlocale(LC_ALL, "Russian");
    srand(time(NULL));

    int num_experiments = 3;

    for (int exp = 1; exp <= num_experiments; exp++)
    {
        std::cout << "Эксперимент #" << exp << " начался..." << std::endl;
        
        std::vector<std::vector<int>> matrixA(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
        std::vector<std::vector<int>> matrixB(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
        std::vector<std::vector<int>> result_matrix(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
        
        OperationStats stats;

        fill_matrix_with_random(matrixA);
        fill_matrix_with_random(matrixB);

        std::cout << "  Запись исходных матриц в input.txt..." << std::endl;
        write_matrices_to_file(matrixA, matrixB, exp);

        std::cout << "  Умножение матриц..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        multiply_matrices(matrixA, matrixB, result_matrix, stats);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "  Запись результата в result.txt..." << std::endl;
        write_result_to_file(result_matrix, elapsed.count(), stats, exp);

        write_data_to_file(exp, elapsed.count(), stats);
        
    }
    
    std::cout << "Все эксперименты завершены!" << std::endl;
    return 0;
}