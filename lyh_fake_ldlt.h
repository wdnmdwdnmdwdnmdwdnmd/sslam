#ifndef LYH_FAKE_LDLT_H
#define LYH_FAKE_LDLT_H

#include "lyh-cholesky.h"
#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <atomic>

namespace lyh {

// Global statistics
static std::atomic<int> g_neon_call_count{0};
static std::atomic<double> g_neon_total_time{0.0};

/**
 * @brief A LDLT-like wrapper implemented using LLT
 *        Designed to match g2o Eigen::LDLT interface semantics
 */
class FakeLDLT {
public:
  using MatrixType = Eigen::MatrixXd;
  using VectorType = Eigen::VectorXd;

  FakeLDLT() : m_isPositive(false) {}

  /// mimic Eigen::LDLT::compute()
  FakeLDLT& compute(const MatrixType& H) {
    auto start = std::chrono::high_resolution_clock::now();
    
    m_llt.compute(H.cast<float>());
    m_isPositive = m_llt.isInitialized();
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    g_neon_call_count++;
    g_neon_total_time.store(g_neon_total_time.load() + time_ms);
    
    static int print_every = 100;
    if (g_neon_call_count % print_every == 0) {
        double avg_time = g_neon_total_time.load() / g_neon_call_count.load();
        std::cout << "[NEON LLT] Calls: " << g_neon_call_count.load() 
                  << ", Matrix: " << H.rows() << "x" << H.cols()
                  << ", Last: " << time_ms << "ms"
                  << ", Avg: " << avg_time << "ms"
                  << std::endl;
    }
    
    return *this;
  }

  /// mimic Eigen::LDLT::solve()
  VectorType solve(const VectorType& b) const {
    assert(m_isPositive && "FakeLDLT::solve() called on non-positive matrix");

    // Cast the double-precision input vector b to a float-precision vector
    Eigen::VectorXf b_float = b.cast<float>();

    // Solve L * y = b (all in float precision)
    Eigen::VectorXf y_float = m_llt.matrixL()
                       .template triangularView<Eigen::Lower>()
                       .solve(b_float);

    // Solve L^T * x = y (all in float precision)
    Eigen::VectorXf x_float = m_llt.matrixL()
                       .transpose()
                       .template triangularView<Eigen::Upper>()
                       .solve(y_float);
    
    // Cast the float-precision result back to double-precision and return
    return x_float.cast<double>();
  }

  /// mimic Eigen::LDLT::isPositive()
  bool isPositive() const {
    return m_isPositive;
  }

private:
  lyh::CholeskyLLT m_llt;
  bool m_isPositive;
};

// Helper function to print final statistics
inline void printNEONStats() {
    int calls = g_neon_call_count.load();
    double total = g_neon_total_time.load();
    std::cout << "\n========================================" << std::endl;
    std::cout << "  NEON LLT Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total calls:     " << calls << std::endl;
    std::cout << "Total time:      " << total << " ms" << std::endl;
    if (calls > 0) {
        std::cout << "Average time:    " << total / calls << " ms/call" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}

} // namespace lyh

#endif // LYH_FAKE_LDLT_H
