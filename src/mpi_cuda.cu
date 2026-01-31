#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <mpi.h>

int N_T = 500;
double T = 1.0, tau = 0;
double d[3] = { 0.0, 0.0, 0.0 };
double all_len[3] = { 1.0, 1.0, 1.0 };
int all_size[3] = { 128, 128, 128 };
int num_blocks[3] = { 1, 1, 1 };
int block_size[3] = { 0, 0, 0 };
int grid_pos[3] = { 0, 0, 0 };
double global_pos[3] = { 0, 0, 0 };
double A, B, C, D;

__device__ double tau_dev = 0;
__device__ double d_dev[3] = { 0.0, 0.0, 0.0 };
__device__ int all_size_dev[3] = { 128, 128, 128 };
__device__ int block_size_dev[3] = { 0, 0, 0 };
__device__ double global_pos_dev[3] = { 0, 0, 0 };
__device__ double A_dev, B_dev, C_dev, D_dev;

int print_arr_size = 0;
int all_print_z = 0;
int print_arr[10];
int print_arr_num[10];

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ int pos(int *p)
{
    return p[0] + p[1] * block_size_dev[0] + p[2] * block_size_dev[0] * block_size_dev[1];
}

__device__ int pos_c(int i, int j, int k)
{
    return i + j * block_size_dev[0] + k * block_size_dev[0] * block_size_dev[1];
}

__device__ double ini_val(int i, int j, int k)
{
    // return sin(A_dev * i) * sin(B_dev * j) * sin(C_dev * k);
    return 0;
}

__device__ double ini_diff_val(int i, int j, int k)
{
    return 0;
}

__device__ double source(int i, int j, int k, int t)
{
    // return 0;
    if (i == all_size_dev[0] / 3 && j == all_size_dev[1] / 3 && k == all_size_dev[2] / 3)
        return all_size_dev[0] * all_size_dev[1] * all_size_dev[2] * sin(t * tau_dev * 100);
    else
        return 0;
    return all_size_dev[0] * all_size_dev[1] * all_size_dev[2] / 3 *
           exp(-1.0 * ((i - all_size_dev[0] / 3) * (i - all_size_dev[0] / 3) +
                       (j - all_size_dev[1] / 3) * (j - all_size_dev[1] / 3) +
                       (k - all_size_dev[2] / 3) * (k - all_size_dev[2] / 3))) *
           sin(t * tau_dev * 100);
}

__device__ double laplace(int *p, double *arr)
{
    double q = -2 * arr[pos_c(p[0], p[1], p[2])];
    double laplace_x = (arr[pos_c(p[0] - 1, p[1], p[2])] + q + arr[pos_c(p[0] + 1, p[1], p[2])]) /
                       (d_dev[0] * d_dev[0]);
    double laplace_y = (arr[pos_c(p[0], p[1] - 1, p[2])] + q + arr[pos_c(p[0], p[1] + 1, p[2])]) /
                       (d_dev[1] * d_dev[1]);
    double laplace_z = (arr[pos_c(p[0], p[1], p[2] - 1)] + q + arr[pos_c(p[0], p[1], p[2] + 1)]) /
                       (d_dev[2] * d_dev[2]);
    return laplace_x + laplace_y + laplace_z;
}

__device__ int satmod(int a, int b)
{
    if (a == -1)
        return b - 2;
    if (a == b)
        return 1;
    return a;
}

int pos_c_cpu(int i, int j, int k)
{
    return i + j * block_size[0] + k * block_size[0] * block_size[1];
}

int from_cord_to_rank(int *p)
{
    for (int i = 0; i < 3; i++) {
        if (p[i] == -1)
            p[i] = num_blocks[i] - 1;
        if (p[i] == num_blocks[i])
            p[i] = 0;
    }

    return p[0] + p[1] * num_blocks[0] + p[2] * num_blocks[0] * num_blocks[1];
}

float ReverseFloat(const float inFloat)
{
    float retVal;
    char *floatToConvert = (char *)&inFloat;
    char *returnFloat = (char *)&retVal;

    // swap the bytes into a temporary buffer
    returnFloat[0] = floatToConvert[3];
    returnFloat[1] = floatToConvert[2];
    returnFloat[2] = floatToConvert[1];
    returnFloat[3] = floatToConvert[0];

    return retVal;
}

void write_to_file(double *in, int t, int rank, int size)
{
    int init_seek = 0;
    FILE *fptr = NULL;
    char out_name[100];

    sprintf(out_name, "plot/out_%d.vtk", t);
    if (rank == 0) {
        fptr = fopen(out_name, "wb");
        fprintf(fptr, "# vtk DataFile Version 2.0\n");
        fprintf(fptr, "Wave\n");
        fprintf(fptr, "BINARY\n");
        fprintf(fptr, "DATASET STRUCTURED_POINTS\n");
        fprintf(fptr, "DIMENSIONS %d %d %d\n", all_size[0], all_size[1], all_print_z);
        fprintf(fptr, "ASPECT_RATIO %f %f %f\n", d[0], d[1], all_len[2] / all_print_z);
        fprintf(fptr, "ORIGIN %d %d %d\n", 0, 0, 0);
        fprintf(fptr, "POINT_DATA %d\n", all_size[0] * all_size[1] * all_print_z);
        fprintf(fptr, "SCALARS value_gpu float 1\n");
        fprintf(fptr, "LOOKUP_TABLE default\n");
        init_seek = ftell(fptr);
        fclose(fptr);
    }

    MPI_Bcast(&init_seek, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int proc = 0; proc < size; proc++) {
        if (rank == proc) {
            fptr = fopen(out_name, "r+b");
            for (int p = 0; p < print_arr_size; p++) {
                for (int j = 1; j < block_size[1] - 1; j++) {
                    fseek(fptr,
                          sizeof(float) * (global_pos[0] + all_size[0] * (global_pos[1] + j - 1) +
                                           all_size[0] * all_size[1] * print_arr_num[p]) +
                              init_seek,
                          SEEK_SET);
                    for (int i = 1; i < block_size[0] - 1; i++) {
                        float tmp1 = ReverseFloat(in[pos_c_cpu(i, j, print_arr[p])]);
                        fwrite(&tmp1, sizeof(float), 1, fptr);
                    }
                }
            }
            fclose(fptr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

double curr_time;
void time_print(const char *name, int rank)
{
    (void)name;
    (void)rank;

#ifdef log_time
    if (rank == 0)
        if (name != NULL)
            printf("Spend %s time:%fs\n", name, MPI_Wtime() - curr_time);
        else
            curr_time = MPI_Wtime();
#endif
}

__global__ void init(double *in)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    if (idx < block_size_dev[0] && idy < block_size_dev[1] && idz < block_size_dev[2]) {
        int p[3] = { idx, idy, idz };
        in[pos(p)] = ini_val(satmod(global_pos_dev[0] + idx - 1, all_size_dev[0]),
                             satmod(global_pos_dev[1] + idy - 1, all_size_dev[1]),
                             satmod(global_pos_dev[2] + idz - 1, all_size_dev[2]));
    }
}

__global__ void first_step(double *in, double *inp)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    if (idx < block_size_dev[0] - 1 && idy < block_size_dev[1] - 1 && idz < block_size_dev[2] - 1 &&
        idx > 0 && idy > 0 && idz > 0) {
        int p[3] = { idx, idy, idz };
        int mark = 1;

        for (int m = 0; m < 3; m++) {
            if ((global_pos_dev[m] + p[m] - 1 == 0) ||
                (global_pos_dev[m] + p[m] - 1 == all_size_dev[m] - 1)) {
                in[pos(p)] = 0;
                mark = 0;
            }
        }

        if (mark)
            in[pos(p)] = inp[pos(p)] +
                         tau_dev * ini_diff_val(global_pos_dev[0] + idx - 1,
                                                global_pos_dev[1] + idy - 1,
                                                global_pos_dev[2] + idz - 1) +
                         0.5 * tau_dev * tau_dev * laplace(p, inp) +
                         tau_dev * source(global_pos_dev[0] + idx - 1, global_pos_dev[1] + idy - 1,
                                          global_pos_dev[2] + idz - 1, 1);
    }
}

__global__ void solve(double *in, double *inp, int t)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    if (idx < block_size_dev[0] - 1 && idy < block_size_dev[1] - 1 && idz < block_size_dev[2] - 1 &&
        idx > 0 && idy > 0 && idz > 0) {
        int p[3] = { idx, idy, idz };
        int mark = 1;

        for (int m = 0; m < 3; m++) {
            if ((global_pos_dev[m] + p[m] - 1 == 0) ||
                (global_pos_dev[m] + p[m] - 1 == all_size_dev[m] - 1)) {
                in[pos(p)] = 0;
                mark = 0;
            }
        }

        if (mark)
            in[pos(p)] = 2 * inp[pos(p)] - in[pos(p)] + tau_dev * tau_dev * laplace(p, inp) +
                         tau_dev * tau_dev *
                             source(global_pos_dev[0] + idx - 1, global_pos_dev[1] + idy - 1,
                                    global_pos_dev[2] + idz - 1, t);
    }
}

__global__ void border_load(double *out, double *in, int max_k, int max_f, int tmp, int j)
{
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int f = threadIdx.y + blockDim.y * blockIdx.y;

    if (k < max_k && f < max_f && k > 0 && f > 0) {
        int tmp_p[3];
        tmp_p[j] = tmp;
        tmp_p[(j + 1) % 3] = k;
        tmp_p[(j + 2) % 3] = f;
        out[f + k * block_size_dev[(j + 2) % 3]] = in[pos(tmp_p)];
    }
}

__global__ void border_write(double *out, double *in, int max_k, int max_f, int tmp, int j)
{
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int f = threadIdx.y + blockDim.y * blockIdx.y;

    if (k < max_k && f < max_f && k > 0 && f > 0) {
        int tmp_p[3];
        tmp_p[j] = tmp;
        tmp_p[(j + 1) % 3] = k;
        tmp_p[(j + 2) % 3] = f;
        in[pos(tmp_p)] = out[f + k * block_size_dev[(j + 2) % 3]];
    }
}

int main(int argc, char **argv)
{
    int rank, proc;
    MPI_Request request[12];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc);

    if ((proc & (proc - 1))) {
        if (rank == 0) {
            printf("Processors count not pow of two.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (argc > 1) {
        all_size[0] = all_size[1] = all_size[2] = atoi(argv[1]);
    }

    double time_start = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        time_start = MPI_Wtime();
        printf("Problem size:%d  processors:%d\n", all_size[0], proc);
    }

    time_print(NULL, rank);
    for (int i = 0; i < 3; i++) {
        d[i] = all_len[i] / (all_size[i] - 1);
    }

    tau = fmin(d[0], fmin(d[1], d[2])) / 2;

    A = 2.0 * M_PI / (all_size[0] - 1), B = 2.0 * M_PI / (all_size[1] - 1),
    C = 2.0 * M_PI / (all_size[2] - 1),
    D = sqrt(A * A / (d[0] * d[0]) + B * B / (d[1] * d[1]) + C * C / (d[2] * d[2]));

    int tmp = proc, curr = 0;
    while (tmp != 1) {
        num_blocks[curr] *= 2;
        curr = (curr + 1) % 3;
        tmp /= 2;
    }

    grid_pos[2] = rank / (num_blocks[0] * num_blocks[1]);
    tmp = rank % (num_blocks[0] * num_blocks[1]);
    grid_pos[1] = tmp / num_blocks[0];
    grid_pos[0] = tmp % num_blocks[0];

    for (int i = 0; i < 3; i++) {
        block_size[i] = all_size[i] / num_blocks[i] + 2;
        global_pos[i] = (block_size[i] - 2) * grid_pos[i];
    }

#ifdef data_write
    for (float z_step = 1.0; z_step < all_size[2] - 1; z_step += (all_size[2] - 3.0) / 3) {
        if ((global_pos[2] <= z_step) && (z_step < global_pos[2] + block_size[2] - 2)) {
            print_arr[print_arr_size] = z_step - global_pos[2] + 1;
            print_arr_num[print_arr_size] = all_print_z;
            print_arr_size++;
        }
        all_print_z++;
    }
#endif

    double *arr = (double *)malloc(sizeof(double) * block_size[0] * block_size[1] * block_size[2]);

    double *arr_dev[2];
    gpuErrchk(cudaMalloc((void **)&arr_dev[0],
                         sizeof(double) * block_size[0] * block_size[1] * block_size[2]));
    gpuErrchk(cudaMalloc((void **)&arr_dev[1],
                         sizeof(double) * block_size[0] * block_size[1] * block_size[2]));

    double *in_buffers[6];
    double *out_buffers[6];
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 2; i++) {
            in_buffers[i + 2 * j] = (double *)malloc(sizeof(double) * block_size[(j + 1) % 3] *
                                                     block_size[(j + 2) % 3]);
            out_buffers[i + 2 * j] = (double *)malloc(sizeof(double) * block_size[(j + 1) % 3] *
                                                      block_size[(j + 2) % 3]);
        }
    }

    double *buffer_dev;
    int max_buf_size = max(block_size[0], max(block_size[1], block_size[2]));
    gpuErrchk(cudaMalloc((void **)&buffer_dev, sizeof(double) * max_buf_size * max_buf_size));

    gpuErrchk(cudaMemcpyToSymbol(all_size_dev, all_size, sizeof(int) * 3));
    gpuErrchk(cudaMemcpyToSymbol(block_size_dev, block_size, sizeof(int) * 3));
    gpuErrchk(cudaMemcpyToSymbol(d_dev, d, sizeof(double) * 3));
    gpuErrchk(cudaMemcpyToSymbol(global_pos_dev, global_pos, sizeof(double) * 3));
    gpuErrchk(cudaMemcpyToSymbol(tau_dev, &tau, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(A_dev, &A, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(B_dev, &B, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(C_dev, &C, sizeof(double)));
    gpuErrchk(cudaMemcpyToSymbol(D_dev, &D, sizeof(double)));
    time_print("init", rank);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(block_size[0] / threadsPerBlock.x + 1, block_size[1] / threadsPerBlock.y + 1,
                   block_size[2] / threadsPerBlock.z + 1);

    time_print(NULL, rank);
    init<<<numBlocks, threadsPerBlock>>>(arr_dev[0]);
    time_print("0 layer", rank);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    time_print(NULL, rank);
    first_step<<<numBlocks, threadsPerBlock>>>(arr_dev[1], arr_dev[0]);
    time_print("1 layer", rank);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

#ifdef data_write
    time_print(NULL, rank);
    gpuErrchk(cudaMemcpy(arr, arr_dev[0],
                         sizeof(double) * block_size[0] * block_size[1] * block_size[2],
                         cudaMemcpyDeviceToHost));
    write_to_file(arr, 0, rank, proc);
    gpuErrchk(cudaMemcpy(arr, arr_dev[1],
                         sizeof(double) * block_size[0] * block_size[1] * block_size[2],
                         cudaMemcpyDeviceToHost));
    write_to_file(arr, 1, rank, proc);
    time_print("copy and write 0 and 1 layers", rank);
#endif

    char name[100];

    for (int t = 2; t < N_T; t++) {
        time_print(NULL, rank);
        // Receive data
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 2; i++) {
                int tmp_grid[3] = { grid_pos[0], grid_pos[1], grid_pos[2] };
                tmp_grid[j] += -1 + 2 * i;
                MPI_Irecv(in_buffers[i + 2 * j], block_size[(j + 1) % 3] * block_size[(j + 2) % 3],
                          MPI_DOUBLE, from_cord_to_rank(tmp_grid), i + 2 * j, MPI_COMM_WORLD,
                          &request[i + 2 * j]);
            }
        }

        // Send data
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 2; i++) {
                int tmp = (1 + (grid_pos[j] == 0)) * (i == 0) +
                          (block_size[j] - 2 - (grid_pos[j] == (num_blocks[j] - 1))) * i;

                dim3 threadsPerBlock_tmp(8, 8);
                dim3 numBlocks_tmp((block_size[(j + 1) % 3] - 1) / threadsPerBlock.x + 1,
                                   (block_size[(j + 2) % 3] - 1) / threadsPerBlock.y + 1);

                border_load<<<numBlocks_tmp, threadsPerBlock_tmp>>>(
                    buffer_dev, arr_dev[(t + 1) % 2], block_size[(j + 1) % 3] - 1,
                    block_size[(j + 2) % 3] - 1, tmp, j);
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());

                gpuErrchk(
                    cudaMemcpy(out_buffers[i + 2 * j], buffer_dev,
                               sizeof(double) * block_size[(j + 1) % 3] * block_size[(j + 2) % 3],
                               cudaMemcpyDeviceToHost));

                int tmp_grid[3] = { grid_pos[0], grid_pos[1], grid_pos[2] };
                tmp_grid[j] += -1 + 2 * i;
                MPI_Isend(out_buffers[i + 2 * j], block_size[(j + 1) % 3] * block_size[(j + 2) % 3],
                          MPI_DOUBLE, from_cord_to_rank(tmp_grid), 1 - i + 2 * j, MPI_COMM_WORLD,
                          &request[i + 2 * j + 6]);
            }
        }

        // Write data
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 2; i++) {
                MPI_Wait(&request[i + 2 * j], MPI_STATUS_IGNORE);

                gpuErrchk(
                    cudaMemcpy(buffer_dev, in_buffers[i + 2 * j],
                               sizeof(double) * block_size[(j + 1) % 3] * block_size[(j + 2) % 3],
                               cudaMemcpyHostToDevice));

                int tmp = (block_size[j] - 1) * i;

                dim3 threadsPerBlock_tmp(8, 8);
                dim3 numBlocks_tmp((block_size[(j + 1) % 3] - 1) / threadsPerBlock.x + 1,
                                   (block_size[(j + 2) % 3] - 1) / threadsPerBlock.y + 1);

                border_write<<<numBlocks_tmp, threadsPerBlock_tmp>>>(
                    buffer_dev, arr_dev[(t + 1) % 2], block_size[(j + 1) % 3] - 1,
                    block_size[(j + 2) % 3] - 1, tmp, j);
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
        }

        for (int i = 6; i < 12; i++) {
            MPI_Wait(&request[i], MPI_STATUS_IGNORE);
        }
        time_print("all communication and copy border", rank);

        // Solve
        time_print(NULL, rank);
        solve<<<numBlocks, threadsPerBlock>>>(arr_dev[t % 2], arr_dev[(t + 1) % 2], t);
        sprintf(name, "%d layer", t);
        time_print(name, rank);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

#ifdef data_write
        time_print(NULL, rank);
        gpuErrchk(cudaMemcpy(arr, arr_dev[t % 2],
                             sizeof(double) * block_size[0] * block_size[1] * block_size[2],
                             cudaMemcpyDeviceToHost));
        write_to_file(arr, t, rank, proc);
        sprintf(name, "copy and write %d layer", t);
        time_print(name, rank);
#endif
    }

    time_print(NULL, rank);
    for (int i = 0; i < 6; i++) {
        free(in_buffers[i]);
        free(out_buffers[i]);
    }
    free(arr);

    cudaFree(buffer_dev);
    cudaFree(arr_dev[0]);
    cudaFree(arr_dev[1]);
    time_print("clear", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Spend time:%f\n", MPI_Wtime() - time_start);
    }

    MPI_Finalize();
    return 0;
}
