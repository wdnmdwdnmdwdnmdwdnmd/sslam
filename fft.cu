#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <cmath>
#include <complex>

#define N 1024  // FFT点数
#define PI 3.14159265358979323846

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// 位反转函数
__device__ unsigned int reverseBits(unsigned int num, int bits) {
    unsigned int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (num & 1);
        num >>= 1;
    }
    return result;
}

// 位反转排序kernel
__global__ void bitReverseKernel(cufftComplex* data, cufftComplex* output, int n, int bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int reversedIdx = reverseBits(idx, bits);
        output[reversedIdx] = data[idx];
    }
}

// FFT butterfly操作kernel (Cooley-Tukey算法)
__global__ void fftKernel(cufftComplex* data, int stage, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = 1 << stage;  // 当前阶段的组大小
    int group = idx / (m / 2);
    int pos = idx % (m / 2);
    int k = group * m + pos;
    
    if (k < n) {
        // 计算旋转因子 W_N^k = e^(-2*pi*i*k/N)
        float angle = -2.0f * PI * pos / m;
        cufftComplex w;
        w.x = cosf(angle);  // 实部
        w.y = sinf(angle);  // 虚部
        
        int partner = k + m / 2;
        if (partner < n) {
            // 获取蝶形操作的两个输入
            cufftComplex u = data[k];
            cufftComplex t = data[partner];
            
            // 复数乘法: t = t * w
            cufftComplex tw;
            tw.x = t.x * w.x - t.y * w.y;
            tw.y = t.x * w.y + t.y * w.x;
            
            // 蝶形操作
            data[k].x = u.x + tw.x;
            data[k].y = u.y + tw.y;
            data[partner].x = u.x - tw.x;
            data[partner].y = u.y - tw.y;
        }
    }
}

// 主机端FFT函数（使用自定义kernel）
void customFFT(cufftComplex* h_input, cufftComplex* h_output) {
    cufftComplex *d_data, *d_temp;
    size_t size = N * sizeof(cufftComplex);
    
    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_temp, size));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_data, h_input, size, cudaMemcpyHostToDevice));
    
    // 计算需要的比特数
    int bits = (int)log2(N);
    
    // 位反转排序
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    bitReverseKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_temp, N, bits);
    CUDA_CHECK(cudaMemcpy(d_data, d_temp, size, cudaMemcpyDeviceToDevice));
    
    // FFT迭代
    for (int stage = 1; stage <= bits; stage++) {
        int m = 1 << stage;
        int numButterflies = N / 2;
        blocksPerGrid = (numButterflies + threadsPerBlock - 1) / threadsPerBlock;
        fftKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, stage, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_output, d_data, size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_temp));
}

// 使用cuFFT库的版本（推荐）
void cuFFTVersion(cufftComplex* h_input, cufftComplex* h_output) {
    cufftComplex* d_data;
    size_t size = N * sizeof(cufftComplex);
    
    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_data, h_input, size, cudaMemcpyHostToDevice));
    
    // 创建cuFFT计划
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    
    // 执行FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_output, d_data, size, cudaMemcpyDeviceToHost));
    
    // 清理
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_data));
}

// 生成测试信号：混合正弦波
void generateTestSignal(cufftComplex* signal) {
    for (int i = 0; i < N; i++) {
        // 混合10Hz和50Hz的正弦波
        float t = (float)i / N;
        signal[i].x = sinf(2.0f * PI * 10.0f * t) + 0.5f * sinf(2.0f * PI * 50.0f * t);
        signal[i].y = 0.0f;  // 虚部为0
    }
}

// 打印部分结果
void printResults(const char* title, cufftComplex* data, int numSamples = 10) {
    std::cout << "\n" << title << ":\n";
    for (int i = 0; i < numSamples && i < N; i++) {
        float magnitude = sqrtf(data[i].x * data[i].x + data[i].y * data[i].y);
        float phase = atan2f(data[i].y, data[i].x);
        std::cout << "  [" << i << "] 幅值: " << magnitude 
                  << ", 相位: " << phase << " rad\n";
    }
}

int main() {
    std::cout << "=== 1024点FFT CUDA程序 ===\n";
    std::cout << "FFT点数: " << N << "\n";
    
    // 分配主机内存
    cufftComplex* h_input = new cufftComplex[N];
    cufftComplex* h_output_custom = new cufftComplex[N];
    cufftComplex* h_output_cufft = new cufftComplex[N];
    
    // 生成测试信号
    generateTestSignal(h_input);
    std::cout << "\n测试信号: 10Hz + 0.5*50Hz 正弦波混合\n";
    
    // 方法1: 自定义FFT实现
    std::cout << "\n--- 使用自定义CUDA kernel ---\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    customFFT(h_input, h_output_custom);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float custom_time = 0;
    cudaEventElapsedTime(&custom_time, start, stop);
    std::cout << "执行时间: " << custom_time << " ms\n";
    printResults("自定义FFT结果（前10个点）", h_output_custom);
    
    // 方法2: cuFFT库实现
    std::cout << "\n--- 使用cuFFT库 ---\n";
    cudaEventRecord(start);
    cuFFTVersion(h_input, h_output_cufft);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cufft_time = 0;
    cudaEventElapsedTime(&cufft_time, start, stop);
    std::cout << "执行时间: " << cufft_time << " ms\n";
    printResults("cuFFT结果（前10个点）", h_output_cufft);
    
    // 验证两种方法的结果是否一致
    std::cout << "\n--- 结果验证 ---\n";
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff_real = fabsf(h_output_custom[i].x - h_output_cufft[i].x);
        float diff_imag = fabsf(h_output_custom[i].y - h_output_cufft[i].y);
        max_diff = fmaxf(max_diff, fmaxf(diff_real, diff_imag));
    }
    std::cout << "最大差异: " << max_diff << "\n";
    if (max_diff < 0.01f) {
        std::cout << "✓ 两种方法结果一致！\n";
    } else {
        std::cout << "✗ 结果存在差异\n";
    }
    
    // 清理
    delete[] h_input;
    delete[] h_output_custom;
    delete[] h_output_cufft;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\n程序执行完毕！\n";
    return 0;
}
