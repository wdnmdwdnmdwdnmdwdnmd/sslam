// 示例：用 g2o 实现一个简化版的 VIO 优化器
// 目的：把前面“g2o 图”的中文解释，落成一份可以对照阅读的 C++ 代码。
//
// 注意：
// 1. 这是一个“教学示例文件”，未接入当前 xrslam 的真实类型，也没有加入 CMake 构建。
// 2. 真正接入工程时，需要：
//    - 把这里的 DummyFrame / DummyMapDatabase 换成你的 database::Frame / MapDataBase 等类型；
//    - 把 addReprojectionEdges / addImuEdges 的伪实现，改成用你真实的观测和预积分数据。

#include <memory>
#include <vector>

// g2o 相关头文件（示例中使用最常见的一套配置）
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>  // VertexSE3Expmap / EdgeProjectXYZ2UV 等

namespace slam {

// ===================== 一些“占位用”的示例类型 =====================
// 这里只是为了让示例能自洽，不与现有工程冲突。

// 简化版“帧状态”：只保留一个 SE3 位姿，方便示例
struct DummyFrameState {
  g2o::SE3Quat pose;  // 位姿：四元数 + 平移
};

// 简化版“帧”对象：实际工程里应该替换为 database::PrimaryFrame
struct DummyFrame {
  using Ptr = std::shared_ptr<DummyFrame>;

  DummyFrameState state;

  DummyFrameState &get_state() { return state; }
  const DummyFrameState &get_state() const { return state; }
};

// 简化版“外点信息”占位
struct DummyOutlierInformation {
  using Ptr = std::shared_ptr<DummyOutlierInformation>;
};

// 简化版“地图数据库”：仅提供一个接口返回滑窗内所有帧
struct DummyMapDatabase {
  using Ptr = std::shared_ptr<DummyMapDatabase>;

  // 在真实工程中，这里应该返回 sliding window 内的若干 weak_ptr<Frame>
  std::vector<DummyFrame::Ptr> sliding_window_key_frames_until_frame(const DummyFrame::Ptr &/*current_frame*/) {
    return frames_;  // 示例直接返回内部保存的所有帧
  }

  std::vector<DummyFrame::Ptr> frames_;
};

// ===================== g2o 顶点定义（仅示例位姿顶点） =====================

// 顶点：位姿（SE3）；这里直接用 g2o 内置的 VertexSE3Expmap
using PoseVertex = g2o::VertexSE3Expmap;

// ===================== 示例：基于 g2o 的 VIO 优化器 =====================

class G2oVioEstimatorExample {
 public:
  explicit G2oVioEstimatorExample(const std::shared_ptr<DummyMapDatabase> &map_database)
      : map_database_(map_database) {}

  // 对应我们之前讨论的：
  // vio_estimator_->optimize(frame, mapping_outliers, max_iterations);
  //
  // 这里的参数都用 DummyXxx 做占位，方便把结构看清楚：
  // - current_frame: 当前关键帧
  // - mapping_outliers: 输出的外点信息（这里不做真实填充，只展示流程）
  // - max_iterations: 最大迭代次数
  void optimize(const DummyFrame::Ptr &current_frame,
                std::vector<DummyOutlierInformation::Ptr> &mapping_outliers,
                int max_iterations) {
    // 1. 创建 g2o 优化器，并设置求解器（LM + CSparse）
    g2o::SparseOptimizer optimizer;
    setupSolver(optimizer);  // 用中文注释详细解释见下面函数

    // 2. 从“地图数据库”中取出滑动窗口内所有关键帧
    const auto sliding_window_frames =
        map_database_->sliding_window_key_frames_until_frame(current_frame);

    // 3. 为每个帧添加“位姿顶点”（VertexPose）
    //
    // 对照解释：
    // - 之前我们说：每一帧 Fk 有一个 pose 顶点 VertexPose_k
    // - 这里就是真的给每一帧添加一个 PoseVertex（g2o::VertexSE3Expmap）
    int next_vertex_id = 0;
    for (const auto &f : sliding_window_frames) {
      addPoseVertex(optimizer, f, next_vertex_id);
    }

    // 4. 添加各类约束（边）：
    //   - 视觉重投影约束（reprojection edges）
    //   - IMU 预积分约束（IMU preintegration edges）
    //
    // 这里由于没有接入真实观测数据，只写成“结构示例 + 中文注释”。
    for (const auto &f : sliding_window_frames) {
      addReprojectionEdges(optimizer, f, mapping_outliers);
      addImuEdges(optimizer, f);
    }

    // 5. 添加先验约束（prior edges），例如第一帧位姿的高斯先验
    addPriors(optimizer, sliding_window_frames);

    // 6. 运行非线性优化（LM / Gauss-Newton）
    optimizer.initializeOptimization();
    optimizer.optimize(max_iterations);

    // 7. 从 g2o 顶点中取回优化后的位姿，写回 DummyFrame
    for (const auto &f : sliding_window_frames) {
      updateFrameStateFromVertices(optimizer, f);
    }

    // 8. 根据残差 / chi2 做外点判定，填充 mapping_outliers
    collectOutliers(optimizer, mapping_outliers);
  }

 private:
  // ----------------- 1. 设置求解器（LM + CSparse） -----------------
  //
  // 对应中文解释：
  // - g2o 需要一个“线性求解器 + 块求解器 + 优化算法”的组合
  // - 这里选：CSparse 线性求解器 + BlockSolver_6_3 + Levenberg-Marquardt
  static void setupSolver(g2o::SparseOptimizer &optimizer) {
    using BlockSolverType = g2o::BlockSolver_6_3;
    using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;

    auto linear_solver = std::make_unique<LinearSolverType>();
    auto block_solver = std::make_unique<BlockSolverType>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    optimizer.setAlgorithm(algorithm);
  }

  // ----------------- 2. 为每个帧添加位姿顶点 -----------------
  //
  // - frame: 当前帧（DummyFrame 占位）
  // - next_vertex_id: 顶点 id 递增计数器
  static void addPoseVertex(g2o::SparseOptimizer &optimizer,
                            const DummyFrame::Ptr &frame,
                            int &next_vertex_id) {
    auto *v = new PoseVertex();

    // 从 DummyFrameState 里取出初始位姿，设置到 g2o 顶点
    v->setId(next_vertex_id++);
    v->setEstimate(frame->get_state().pose);

    // 是否固定第一帧位姿，可根据需要在外面控制
    // 这里默认不固定
    v->setFixed(false);

    optimizer.addVertex(v);
  }

  // ----------------- 3. 添加视觉重投影约束（占位） -----------------
  //
  // 真实工程里：
  // - 这里应该遍历该帧上所有特征观测；
  // - 对每个特征，找到对应的 3D 地图点；
  // - 构造 EdgeProjectXYZ2UV（或自定义重投影边），连上 PoseVertex 和 PointVertex；
  // - 设置观测像素 / 信息矩阵 / 鲁棒核等。
  static void addReprojectionEdges(g2o::SparseOptimizer &/*optimizer*/,
                                   const DummyFrame::Ptr &/*frame*/,
                                   std::vector<DummyOutlierInformation::Ptr> &/*mapping_outliers*/) {
    // 示例文件不接入真实观测，这里只保留中文说明。
  }

  // ----------------- 4. 添加 IMU 预积分约束（占位） -----------------
  //
  // 真实工程里：
  // - 需要有 IMU 预积分结果（ΔR, Δv, Δp 及协方差）；
  // - 为 (k, k+1) 两帧的 pose / vel_bias 顶点添加一个多维 IMU 边；
  // - 误差形式通常参考 MSCKF / VINS 相关论文或已有实现。
  static void addImuEdges(g2o::SparseOptimizer &/*optimizer*/,
                          const DummyFrame::Ptr &/*frame*/) {
    // 示例文件不接入真实 IMU，这里只保留中文说明。
  }

  // ----------------- 5. 添加先验约束（占位） -----------------
  //
  // 真实工程里：
  // - 通常会对第一帧的位姿（pose）、速度+bias 施加高斯先验；
  // - 也可以对重力方向等施加先验；
  // - 在 g2o 中表现为“Prior Edge”，连接到对应的顶点上。
  static void addPriors(g2o::SparseOptimizer &/*optimizer*/,
                        const std::vector<DummyFrame::Ptr> &/*frames*/) {
    // 示例文件不接入真实先验，这里只保留中文说明。
  }

  // ----------------- 6. 从 g2o 顶点回写位姿到 Frame -----------------
  //
  // 思路：
  // - 按顶点 id 找到对应的 PoseVertex；
  // - 取出优化后的估计值，写回 DummyFrameState。
  static void updateFrameStateFromVertices(g2o::SparseOptimizer &optimizer,
                                           const DummyFrame::Ptr &frame) {
    // 示例里假设“顶点 id 与 frame 在 sliding_window_frames 的顺序一致”，
    // 实际工程中应建立一个 map<Frame*, vertexId> / map<vertexId, Frame*>。
    //
    // 这里为了不引入额外容器，只演示“查找第一个 PoseVertex 并写回”：
    for (const auto &it : optimizer.vertices()) {
      auto *v = dynamic_cast<PoseVertex *>(it.second);
      if (!v) continue;

      frame->get_state().pose = v->estimate();
      break;
    }
  }

  // ----------------- 7. 根据残差 / chi2 标记外点（占位） -----------------
  //
  // 真实工程里：
  // - 遍历所有重投影边 / IMU 边；
  // - 计算 chi2() 或误差范数；
  // - 超过阈值的观测，创建 OutlierInformation，收集到 mapping_outliers。
  static void collectOutliers(g2o::SparseOptimizer &/*optimizer*/,
                              std::vector<DummyOutlierInformation::Ptr> &/*mapping_outliers*/) {
    // 示例文件不接入真实外点逻辑，这里只保留中文说明。
  }

 private:
  std::shared_ptr<DummyMapDatabase> map_database_;
};

}  // namespace slam


