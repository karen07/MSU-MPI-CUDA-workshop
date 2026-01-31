# MSU MPI CUDA workshop

This repository contains a 3D wave-equation solver developed as an MSU parallel-programming workshop project.

MPI decomposes the computational domain between processes, while CUDA performs the numerical update inside each process. Boundary data is copied between GPU and host memory around MPI exchanges.

The program can produce VTK snapshots and prints timing information for the main stages of the calculation, so the repository also serves as a compact MPI + CUDA implementation example.

## Описание

Этот репозиторий содержит решатель трехмерного волнового уравнения, разработанный как проект практикума МГУ по параллельному программированию.

MPI распределяет вычислительную область между процессами, а CUDA выполняет численное обновление внутри каждого процесса. При обменах MPI граничные данные копируются между памятью GPU и памятью хоста.

Программа может создавать снимки VTK и выводит время выполнения основных этапов расчета, поэтому репозиторий также служит компактным примером реализации MPI + CUDA.

## Сборка

Нужны CMake, MPI, CUDA Toolkit и совместимый NVIDIA GPU.

```sh
cmake --preset release
cmake --build --preset release
```

Исполняемый файл:

```text
build/release/mpi-cuda-waves
```

## Запуск

Количество MPI-процессов должно быть степенью двойки. Первый необязательный аргумент задает размер сетки по трем пространственным направлениям.

```sh
mkdir -p plot
mpirun -np 4 ./build/release/mpi-cuda-waves 128
```

Программа печатает размер задачи, количество процессов и время выполнения отдельных этапов. VTK-файлы записываются в `plot/`.

## Реализация

Каждый MPI-процесс хранит свой трехмерный блок на GPU. CUDA kernels вычисляют очередной временной слой, а MPI обменивает граничные данные с соседними процессами.

Проект учебный и исследовательский: параметры схемы и распределения вычислений находятся в исходном коде и рассчитаны на эксперименты с параллельной реализацией.
