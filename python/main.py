import numpy as np
import os


def read_matrices_from_file(filename, matrix_size, experiment_num):
    """
    Читает матрицы A и B из input.txt для конкретного размера и номера эксперимента
    """
    with open(filename, 'r') as file:
        content = file.read()
    blocks = content.split('------------------------')

    target_block = None
    for block in blocks:
        if f'Размер матриц: {matrix_size}' in block and f'Эксперимент #{experiment_num}' in block:
            target_block = block
            break

    if target_block is None:
        raise ValueError(
            f"Эксперимент #{experiment_num} для размера {matrix_size} не найден")

    lines = target_block.strip().split('\n')

    matrices = {}
    current_matrix = None
    matrix_data = []

    for line in lines:
        line = line.strip()
        if 'Матрица A:' in line:
            current_matrix = 'A'
            matrix_data = []
        elif 'Матрица B:' in line:
            if current_matrix == 'A':
                matrices['A'] = np.array(matrix_data)
            current_matrix = 'B'
            matrix_data = []
        elif current_matrix and line and not line.startswith(
                'Эксперимент') and not line.startswith('Размер'):
            if line.strip():
                row = [int(x) for x in line.split()]
                matrix_data.append(row)

    if current_matrix == 'B' and matrix_data:
        matrices['B'] = np.array(matrix_data)

    return matrices['A'], matrices['B']


def read_result_from_file(filename, matrix_size, experiment_num):
    """
    Читает результирующую матрицу из result.txt
    """
    with open(filename, 'r') as file:
        content = file.read()

    blocks = content.split('------------------------')

    target_block = None
    for block in blocks:
        if f'Размер матриц: {matrix_size}' in block and f'Эксперимент #{experiment_num}' in block:
            target_block = block
            break

    if target_block is None:
        raise ValueError(
            f"Результат для эксперимента #{experiment_num} размера {matrix_size} не найден")

    lines = target_block.strip().split('\n')

    result_data = []
    in_result = False

    for line in lines:
        line = line.strip()
        if 'Результирующая матрица:' in line:
            in_result = True
            continue
        if in_result and line and not any(x in line for x in
                                          ['Время:', 'Размер:', 'Объем:',
                                           'Эксперимент']):
            if line.strip():
                row = [int(x) for x in line.split()]
                result_data.append(row)

    return np.array(result_data)


def verify_experiment(matrix_size, experiment_num):
    """
    Верифицирует один эксперимент для заданного размера и номера
    """
    print(
        f"\n--- Верификация: размер {matrix_size}, эксперимент #{experiment_num} ---")

    try:
        print(f"Чтение матриц из input.txt...")
        A, B = read_matrices_from_file('input.txt', matrix_size, experiment_num)

        print(f"Размер A: {A.shape}")
        print(f"Первые 3x3 элемента A:")
        print(A[:3, :3])
        print(f"Первые 3x3 элемента B:")
        print(B[:3, :3])

        print(f"\nУмножение через NumPy...")
        import time
        start_time = time.time()
        expected = np.dot(A, B)
        numpy_time = time.time() - start_time
        print(f"NumPy время: {numpy_time:.4f} с")

        print(f"\nЧтение результата из result.txt...")
        cpp_result = read_result_from_file('result.txt', matrix_size,
                                           experiment_num)

        print(f"\nСравнение результатов...")

        print("Проверка первых 3x3 элементов:")
        print("C++ результат:")
        print(cpp_result[:3, :3])
        print("NumPy результат:")
        print(expected[:3, :3])

        if matrix_size <= 500:
            if np.array_equal(cpp_result, expected):
                print("✓ РЕЗУЛЬТАТЫ ПОЛНОСТЬЮ СОВПАДАЮТ!")
                return True
            else:
                print("✗ РЕЗУЛЬТАТЫ НЕ СОВПАДАЮТ!")
                diff = np.abs(cpp_result - expected)
                print(f"Максимальная разница: {np.max(diff)}")
                return False
        else:
            test_positions = [(0, 0), (0, 1), (1, 0), (1, 1),
                              (matrix_size - 1, matrix_size - 1)]
            all_match = True
            for i, j in test_positions:
                if i < matrix_size and j < matrix_size:
                    if cpp_result[i, j] == expected[i, j]:
                        print(
                            f"  ✓ Элемент [{i},{j}]: {cpp_result[i, j]} = {expected[i, j]}")
                    else:
                        print(
                            f"  ✗ Элемент [{i},{j}]: {cpp_result[i, j]} ≠ {expected[i, j]}")
                        all_match = False

            if all_match:
                print("✓ ВСЕ ПРОВЕРЕННЫЕ ЭЛЕМЕНТЫ СОВПАДАЮТ!")
            else:
                print("✗ ЕСТЬ РАСХОЖДЕНИЯ!")

            return all_match

    except Exception as e:
        print(f"ОШИБКА: {e}")
        return False


def verify_all():
    """
    Верифицирует все эксперименты для всех размеров
    """
    matrix_sizes = [200, 400, 800, 1200, 1600, 2000]
    experiments_per_size = 3

    print("=" * 70)
    print("ВЕРИФИКАЦИЯ УМНОЖЕНИЯ МАТРИЦ")
    print("=" * 70)

    results = {}

    for size in matrix_sizes:
        print(f"\n{'=' * 50}")
        print(f"РАЗМЕР МАТРИЦ: {size}x{size}")
        print(f"{'=' * 50}")

        size_results = []
        for exp_num in range(1, experiments_per_size + 1):
            success = verify_experiment(size, exp_num)
            size_results.append(success)
            print("-" * 40)

        results[size] = size_results

        success_count = sum(size_results)
        print(
            f"\nИТОГ ДЛЯ РАЗМЕРА {size}: {success_count}/{experiments_per_size} успешно")

    print("\n" + "=" * 70)
    print("ОБЩИЙ ИТОГ ВЕРИФИКАЦИИ:")
    print("=" * 70)

    total_success = 0
    total_experiments = 0

    for size in matrix_sizes:
        success_count = sum(results[size])
        total_success += success_count
        total_experiments += len(results[size])
        print(f"Размер {size:4d}: {success_count}/{len(results[size])} ✓")

    print("-" * 70)
    print(f"ВСЕГО: {total_success}/{total_experiments} экспериментов успешно")

    if total_success == total_experiments:
        print("\n🎉 ВСЕ ЭКСПЕРИМЕНТЫ ПРОШЛИ ВЕРИФИКАЦИЮ!")
    else:
        print(
            f"\n⚠️ {total_experiments - total_success} экспериментов требуют проверки")


if __name__ == "__main__":
    verify_all()