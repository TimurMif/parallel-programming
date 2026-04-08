#include <iostream>
#include <ctime>
#include <fstream>
#include <locale>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

int MATRIX_SIZE = 200;

struct OperationStats
{
    long long multiplications = 0;
    long long additions = 0;
    long long assignments = 0;
    long long total() const { return multiplications + additions + assignments; }
};

void fill_matrix_with_random(std::vector<std::vector<int>>& matrix)
{
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
            matrix[i][j] = rand() % 10;
}

__global__ void matMulKernel(const int* A, const int* B, int* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void multiply_matrices_cuda(const std::vector<std::vector<int>>& A,
                            const std::vector<std::vector<int>>& B,
                            std::vector<std::vector<int>>& result,
                            dim3 blockSize)
{
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);
    int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    std::vector<int> h_A(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<int> h_B(MATRIX_SIZE * MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            h_A[i * MATRIX_SIZE + j] = A[i][j];
            h_B[i * MATRIX_SIZE + j] = B[i][j];
        }

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    dim3 gridSize((MATRIX_SIZE + blockSize.x - 1) / blockSize.x,
                  (MATRIX_SIZE + blockSize.y - 1) / blockSize.y);

    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, MATRIX_SIZE);
    cudaDeviceSynchronize();

    std::vector<int> h_C(MATRIX_SIZE * MATRIX_SIZE);
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < MATRIX_SIZE; ++i)
        for (int j = 0; j < MATRIX_SIZE; ++j)
            result[i][j] = h_C[i * MATRIX_SIZE + j];

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    setlocale(LC_ALL, "Russian");
    srand(static_cast<unsigned>(time(nullptr)));

    std::vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    std::vector<std::pair<int,int>> block_configs = {
        {8,8}, {16,8}, {16,16}, {32,8}, {32,16}, {32,32}
    };
    const int REPEATS = 3;

    std::cout << "=========================================" << std::endl;
    std::cout << "ЛАБОРАТОРНАЯ РАБОТА: Умножение матриц с CUDA" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Размеры матриц: ";
    for (int s : sizes) std::cout << s << " ";
    std::cout << "\nКонфигурации блоков: ";
    for (auto& cfg : block_configs) std::cout << cfg.first << "x" << cfg.second << " ";
    std::cout << "\nКоличество замеров на комбинацию: " << REPEATS << std::endl;
    std::cout << "Результаты будут сохранены в results.csv" << std::endl;
    std::cout << "=========================================" << std::endl;

    std::ofstream csv("results.csv", std::ios::trunc);
    csv << "Размер матрицы,Конфигурация блоков,Потоков в блоке,Время_замер_1_с,Время_замер_2_с,Время_замер_3_с,Среднее_время_с\n";
    csv.close();

    int total_experiments = sizes.size() * block_configs.size();
    int current = 0;

    for (int size : sizes)
    {
        MATRIX_SIZE = size;
        std::vector<std::vector<int>> A(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
        std::vector<std::vector<int>> B(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
        fill_matrix_with_random(A);
        fill_matrix_with_random(B);

        for (auto& cfg : block_configs)
        {
            current++;
            dim3 blockSize(cfg.first, cfg.second);
            int threads_per_block = cfg.first * cfg.second;

            std::cout << "\n[" << current << "/" << total_experiments << "] "
                      << "Размер " << size << "x" << size
                      << ", блок " << cfg.first << "x" << cfg.second
                      << " (" << threads_per_block << " потоков)" << std::endl;

            std::vector<std::vector<int>> C_warm(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
            multiply_matrices_cuda(A, B, C_warm, blockSize);

            std::vector<double> times;
            for (int rep = 0; rep < REPEATS; ++rep)
            {
                std::vector<std::vector<int>> C(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
                auto start = std::chrono::high_resolution_clock::now();
                multiply_matrices_cuda(A, B, C, blockSize);
                auto end = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(end - start).count();
                times.push_back(elapsed);
                std::cout << "  Замер " << (rep+1) << "/" << REPEATS << ": " << std::fixed << std::setprecision(6) << elapsed << " с" << std::endl;
            }

            double avg = 0.0;
            for (double t : times) avg += t;
            avg /= REPEATS;

            std::ofstream csv_out("results.csv", std::ios::app);
            csv_out << size << ","
                    << cfg.first << "x" << cfg.second << ","
                    << threads_per_block << ","
                    << std::fixed << std::setprecision(6) << times[0] << ","
                    << times[1] << ","
                    << times[2] << ","
                    << avg << "\n";
            csv_out.close();

            std::cout << "  Среднее: " << avg << " с" << std::endl;
        }
    }

    std::cout << "\n=========================================" << std::endl;
    std::cout << "Все эксперименты завершены!" << std::endl;
    std::cout << "Результаты сохранены в results.csv" << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}
