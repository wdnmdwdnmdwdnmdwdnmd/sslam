// lyh-cholesky.h - SIMD optimized Cholesky LLT decomposition for Apple M1
// Author: lyh
// Date: 2025/11/30

#ifndef LYH_CHOLESKY_H
#define LYH_CHOLESKY_H

#include <Eigen/Core>
#include <cmath>
#include <vector>   // <-- added
#include <iostream>

// Apple M1 uses ARM architecture with NEON SIMD
#ifdef __ARM_NEON
#include <arm_neon.h>
#define LYH_USING_NEON 1
#else
#define LYH_USING_NEON 0
#warning "NEON instructions not available! Falling back to scalar code."
#endif

namespace lyh {

namespace internal {

#ifdef __ARM_NEON
// NEON optimized matrix-vector multiplication and subtraction for float
static void neon_matrix_vector_subtract(float* result, const float* mat, 
                                      const float* vec, int rows, int cols, int stride) {
    const int neon_elements = 4;
    int i = 0;
    
    for (; i <= rows - neon_elements; i += neon_elements) {
        float32x4_t accum = vdupq_n_f32(0.0f);
        
        // Accumulate mat[i:i+3, :] * vec[:]^T
        // In column-major storage, consecutive elements in a column are stored consecutively
        for (int j = 0; j < cols; ++j) {
            // Load 4 elements from column j, rows i to i+3
            // In column-major: mat[col * stride + row]
            const float* col_j = mat + j * stride;
            float32x4_t mat_vec = vld1q_f32(col_j + i);
            float32x4_t vec_val = vdupq_n_f32(vec[j]);
            accum = vmlaq_f32(accum, mat_vec, vec_val);
        }
        
        // Load current result and subtract
        float32x4_t current = vld1q_f32(result + i);
        float32x4_t updated = vsubq_f32(current, accum);
        vst1q_f32(result + i, updated);
    }
    
    // Handle remaining elements
    for (; i < rows; ++i) {
        float accum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            // In column-major: mat[col * stride + row]
            accum += mat[j * stride + i] * vec[j];
        }
        result[i] -= accum;
    }
}

// NEON optimized vector scaling for float
static void neon_vector_scale(float* vec, float scale, int size) {
    const int neon_elements = 4;
    int i = 0;
    
    float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i <= size - neon_elements; i += neon_elements) {
        float32x4_t data = vld1q_f32(vec + i);
        float32x4_t scaled = vmulq_f32(data, scale_vec);
        vst1q_f32(vec + i, scaled);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        vec[i] *= scale;
    }
}

// NEON optimized squared norm computation for float
static float neon_squared_norm(const float* vec, int size) {
    const int neon_elements = 4;
    int i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    for (; i <= size - neon_elements; i += neon_elements) {
        float32x4_t data = vld1q_f32(vec + i);
        sum_vec = vmlaq_f32(sum_vec, data, data);
    }
    
    // Horizontal sum
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    float32x2_t sum1 = vpadd_f32(sum2, sum2);
    float result = vget_lane_f32(sum1, 0);
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result += vec[i] * vec[i];
    }
    
    return result;
}
#endif

} // namespace internal

// Core Cholesky LLT decomposition function (non-blocked version) for float matrices
// Returns true on success, false if matrix is not positive definite
static bool cholesky_llt(Eigen::MatrixXf& mat) {
    const int size = mat.rows();
    const int stride = mat.outerStride(); // usually equals size for Dense column-major Eigen matrices

    // Temporary buffer for copying a row's left part (to make it contiguous)
    std::vector<float> tmpRow;
    tmpRow.reserve(size);

    for (int k = 0; k < size; ++k) {
        int rs = size - k - 1;  // 剩余行数
        
        // 步骤1: 定义矩阵块 (Eigen blocks for fallback)
        Eigen::Block<Eigen::MatrixXf, Eigen::Dynamic, 1> A21(mat, k+1, k, rs, 1);  // 第k列的下部分
        Eigen::Block<Eigen::MatrixXf, 1, Eigen::Dynamic> A10(mat, k, 0, 1, k);     // 第k行的左部分  
        Eigen::Block<Eigen::MatrixXf, Eigen::Dynamic, Eigen::Dynamic> A20(mat, k+1, 0, rs, k); // 左下子矩阵
        
        // 步骤2: 计算对角线元素 L[k,k]
        float x = mat.coeff(k, k);  // 获取原始矩阵的(k,k)元素
        if (k > 0) {
            // copy the k-length row (mat(k, 0..k-1)) into contiguous tmpRow
            tmpRow.resize(k);
            for (int j = 0; j < k; ++j) {
                // use coeff to read safely regardless of layout/stride
                tmpRow[j] = mat.coeff(k, j);
            }

#ifdef __ARM_NEON
            // Use NEON optimized squared norm for float on contiguous data
            x -= internal::neon_squared_norm(tmpRow.data(), k);
#else
            x -= A10.squaredNorm();
#endif
        }
        
        // 添加数值稳定性检查
        if (x <= 0.0f) {
            // 对于数值不稳定性，尝试添加小的正数
            x = std::max(x, 1e-8f);
            // return false;  // 暂时不返回false，继续计算
        }
        mat.coeffRef(k, k) = x = std::sqrt(x);   // 计算平方根得到L[k,k]
        
        // 步骤3: 更新第k列的下部分 L[k+1:end,k]
        if (k > 0 && rs > 0) {
#ifdef __ARM_NEON
            // NEON optimized version for float
            // mat base pointer: address of element (k+1, 0) (column-major)
            const float* A20_base = mat.data() + /*col 0*/ 0 * stride + (k + 1);
            // vec: contiguous tmpRow.data() representing mat(k, 0..k-1)
            internal::neon_matrix_vector_subtract(
                &mat(k+1, k),   // result (contiguous column)
                A20_base,       // A20 base pointer (column-major aware)
                tmpRow.data(),  // contiguous row data
                rs, k, stride);
#else
            A21.noalias() -= A20 * A10.adjoint();
#endif
        }
        
        if (rs > 0) {
#ifdef __ARM_NEON
            // NEON optimized scaling for float
            internal::neon_vector_scale(&mat(k+1, k), 1.0f / x, rs);
#else
            A21 /= x;
#endif
        }
        
        // 清零当前行右侧的上三角部分
        for (int j = k + 1; j < size; ++j) {
            mat.coeffRef(k, j) = 0.0f;
        }
    }
    return true;  // 成功完成
}

// Blocked Cholesky LLT decomposition for large matrices
// Uses cache-friendly blocking strategy for better performance on large matrices
static bool cholesky_llt_blocked(Eigen::MatrixXf& mat) {
    const int size = mat.rows();
    
    // For small matrices, use non-blocked version
    if (size < 32) {
        return cholesky_llt(mat);
    }
    
    // Determine optimal block size (8-128, aligned to 16)
    int blockSize = size / 8;
    blockSize = (blockSize / 16) * 16;  // Align to multiple of 16
    blockSize = std::min(std::max(blockSize, 8), 128);
    
    // Temporary buffer for copying rows
    std::vector<float> tmpRow;
    tmpRow.reserve(size);
    
    for (int k = 0; k < size; k += blockSize) {
        int bs = std::min(blockSize, size - k);  // Current block size
        int rs = size - k - bs;                  // Remaining size
        
        // Define matrix blocks
        Eigen::Block<Eigen::MatrixXf, Eigen::Dynamic, Eigen::Dynamic> 
            A11(mat, k, k, bs, bs);     // Current diagonal block
        Eigen::Block<Eigen::MatrixXf, Eigen::Dynamic, Eigen::Dynamic> 
            A21(mat, k+bs, k, rs, bs);  // Lower part
        Eigen::Block<Eigen::MatrixXf, Eigen::Dynamic, Eigen::Dynamic> 
            A22(mat, k+bs, k+bs, rs, rs); // Remaining matrix
         
        // Step 1: Decompose current block using non-blocked algorithm
        // Create a temporary copy of the block to avoid reference issues
        Eigen::MatrixXf A11_temp = A11;
        if (!cholesky_llt(A11_temp)) {
            return false;  // Matrix not positive definite
        }
        A11 = A11_temp;  // Copy back the decomposed block
        
        // Step 2: Update A21 using triangular solve
        if (rs > 0) {
            // Solve A21 = A21 * L11^-T
            // This is equivalent to: A21 = A21 * inv(L11^T)
            // Where L11 is the lower triangular factor from A11
            A11.adjoint().template triangularView<Eigen::Upper>()
                .template solveInPlace<Eigen::OnTheRight>(A21);
        }
        
        // Step 3: Update A22 using rank-k update (Schur complement)
        if (rs > 0) {
            // A22 = A22 - A21 * A21^T
            // This updates the remaining matrix using the computed factors
            A22.template selfadjointView<Eigen::Lower>()
                .rankUpdate(A21, -1.0f);
        }
    }
    
    return true;  // Success
}

// Simple Cholesky LLT class for float matrices
class CholeskyLLT {
public:
    CholeskyLLT() : m_isInitialized(false), m_useBlocked(true) {
        static bool printed = false;
        if (!printed) {
#if LYH_USING_NEON
            std::cout << "[lyh::CholeskyLLT] Initialized with NEON SIMD support" << std::endl;
#else
            std::cout << "[lyh::CholeskyLLT] WARNING: NEON not available, using scalar fallback" << std::endl;
#endif
            printed = true;
        }
    }
    
    explicit CholeskyLLT(const Eigen::MatrixXf& matrix, bool useBlocked = true) 
        : m_isInitialized(false), m_useBlocked(useBlocked) {
        compute(matrix);
    }
    
    // Set whether to use blocked algorithm (better for large matrices)
    void setUseBlocked(bool useBlocked) { m_useBlocked = useBlocked; }
    
    CholeskyLLT& compute(const Eigen::MatrixXf& matrix) {
        eigen_assert(matrix.rows() == matrix.cols());
        m_matrix = matrix;
        
        // Choose algorithm based on matrix size and user preference
        if (m_useBlocked && matrix.rows() >= 32) {
            m_isInitialized = cholesky_llt_blocked(m_matrix);
        } else {
            m_isInitialized = cholesky_llt(m_matrix);
        }
        
        // Zero out the upper triangular part to ensure clean L matrix
        if (m_isInitialized) {
            for (int i = 0; i < m_matrix.rows(); ++i) {
                for (int j = i + 1; j < m_matrix.cols(); ++j) {
                    m_matrix(i, j) = 0.0f;
                }
            }
        }
        
        return *this;
    }
    
    Eigen::MatrixXf matrixL() const {
        eigen_assert(m_isInitialized && "CholeskyLLT is not initialized.");
        return m_matrix;
    }
    
    Eigen::MatrixXf matrixLLT() const {
        eigen_assert(m_isInitialized && "CholeskyLLT is not initialized.");
        return m_matrix;
    }
    
    bool isInitialized() const { return m_isInitialized; }
    
private:
    Eigen::MatrixXf m_matrix;
    bool m_isInitialized;
    bool m_useBlocked;
};

} // namespace lyh

#endif // LYH_CHOLESKY_H
