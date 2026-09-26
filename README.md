# portfolio-optimizer-go

本地投资组合构建与再平衡库（纯 Go 标准库，无外部依赖）。输入历史收益序列与约束条件，
输出满足全部硬约束的组合权重、风险指标与可解释的再平衡交易清单。

## 功能概览

- **构建目标**：均值-方差（`ObjectiveMeanVariance`）、风险平价（`ObjectiveRiskParity`）。
- **硬约束**：权重上下限、分组上限、最大杠杆、换手率上限；返回权重严格满足全部硬约束。
- **估计器**：样本协方差（`EstimatorSample`）、Ledoit-Wolf 收缩（`EstimatorShrinkage`）。
- **数据健壮性**：缺失值（NaN/Inf）修复、异常值 winsorize、奇异/近奇异协方差 PSD 修复，
  全部通过 `Diagnostics` 给出可区分诊断。
- **再平衡**：交易成本与换手率上限内生成可行交易，成本取舍可解释，抑制无意义调仓。
- **稳健性诊断**：参数扰动下的权重敏感性、集中度、条件数，识别不稳健配置。
- **确定性**：相同输入与随机种子下输出逐位一致。
- **线程安全**：`Engine` 支持并发构建与查询，结果缓存，异常不留部分写入。

## 模型假设

1. **收益序列**：`Problem.Returns` 为 T×N 简单收益率矩阵（T 期、N 资产），按行时间有序。
   非有限值（NaN/±Inf）以该资产有限观测的列均值填补并计数；超过 ±8σ 的观测被截断
   （winsorize）并计数。
2. **全投资约束**：权重恒满足 `sum(w) = 1`（全额投资），这是内置硬等式。
3. **均值-方差目标**：最大化 `μ'w − ½·λ·w'Σw`，`λ = RiskAversion`（默认 1）。
4. **风险平价目标**：最小化各资产风险贡献 `RC_i = w_i(Σw)_i` 与均值风险贡献的离差
   （scale-free 形式），使各资产对组合波动率的贡献相等。
5. **无风险利率**：Sharpe 比率按 `rf = 0` 计算。
6. **协方差**：样本协方差使用 `T−1` 自由度；收缩估计向常相关目标收缩，强度由
   Ledoit-Wolf 公式估计并钳制在 [0,1]。估计结果若不正定，通过对角抖动投影到 PSD 并
   置 `Diagnostics.ShrunkToPSD = true`（`N > T` 时样本协方差必然奇异，会自动触发）。

## 约束语义

| 字段 | 默认 | 语义 |
|---|---|---|
| `MinWeight` | 0 | 单资产下限；`AllowShort=false` 时被钳制为 ≥0 |
| `MaxWeight` | 1 | 单资产上限 |
| `AllowShort` | false | 是否允许负权重 |
| `MaxLeverage` | 1 | `sum|w| ≤ MaxLeverage`；全投资下必须 ≥1，否则不可行 |
| `MaxTurnover` | 0（禁用） | `sum|w − Current| ≤ MaxTurnover`；启用时必须提供 `Current` |
| `Groups` | 无 | 每组 `sum_{i∈G} w_i ≤ Max`，组名唯一、成员索引须合法 |

约束不可行时（如 `3×MaxWeight < 1`、组上限低于成员下限之和），返回
`StatusInfeasible` 且不携带权重；结构错误（维度不齐、负预算等）返回
`StatusInvalidInput`。

## 求解与终止

求解器为投影梯度法（均值-方差用 FISTA 加速，风险平价用 Armijo 回溯线搜索），
投影算子由 Dykstra 交替投影实现，每个集合投影都是精确二分求解，**全程确定性**。

- `MaxIters`（默认 2000）与 `TimeBudget`（默认不限）控制求解预算。
- 收敛：`StatusOptimal`，`Diagnostics.Converged = true`。
- 预算耗尽：`StatusFeasibleSub`，返回**当前最优可行解**（仍满足全部硬约束），
  并在 `Diagnostics.Message` 中说明——绝不静默返回伪最优。
- 注意：只有 `TimeBudget` 被实际触发时结果才依赖机器时序；不设置时间预算时，
  相同输入逐位可复现。

## 使用方式

```go
res := portfolio.ConstructOnce(portfolio.Problem{
    Returns:   returns, // [][]float64，T×N
    Objective: portfolio.ObjectiveMeanVariance,
    Estimator: portfolio.EstimatorShrinkage,
    Constraints: portfolio.Constraints{
        MaxWeight:   0.3,
        MaxLeverage: 1.0,
        Groups: []portfolio.GroupCap{
            {Name: "equity", Members: []int{0, 1, 2}, Max: 0.6},
        },
    },
    Current:   currentWeights, // 启用换手率约束时必填
    Seed:      42,
})
// res.Weights 严格满足全部硬约束；res.Risk 含期望收益、波动率、Sharpe、
// Herfindahl 集中度、有效资产数、换手率。
```

### 再平衡

```go
out, err := portfolio.Rebalance(portfolio.RebalanceRequest{
    Current:     cur,
    Target:      res.Weights,
    UniformCost: 0.001,  // 或逐资产 CostRate
    MaxTurnover: 0.2,    // 硬上限
    NoTradeBand: 0.005,  // 小于该幅度的交易被抑制
})
// out.Trades 之和为 0（保持全投资）；out.Cost/Benefit/NetBenefit 与
// out.Explanation 完整解释成本取舍。
```

成本取舍规则（可解释）：先投影到 `{sum t = 0, sum|t| ≤ MaxTurnover}`；再抑制低于
`NoTradeBand` 的碎单并按比例恢复零和；最后丢弃"比例成本 × CostAversion ≥ 1"的交易
（成本超过权重距离收益），每一步都记入 `Explanation`。

### 稳健性诊断

```go
rep, err := portfolio.AssessRobustness(problem, 16)
// rep.MaxWeightShift / MeanWeightShift：扰动下权重 L1 漂移
// rep.WeightSensitivity：逐资产敏感性
// rep.Herfindahl / EffectiveN / CondNumber：集中度与病态程度
// rep.Stable：综合判定（漂移 < 0.1 且条件数 < 1e8）
```

扰动为各资产波动率 1% 的高斯噪声，随机源由 `Seed` 固定，报告完全可复现。

### 并发使用

```go
eng := portfolio.NewEngine()
// 多 goroutine 并发调用 eng.Construct(p)：相同问题只计算一次，
// 缓存原子写入，每个调用方拿到私有副本，panic 不会留下部分写入。
```

## 测试

```sh
go test ./... -race
```

覆盖：约束边界（box/组/杠杆/换手）、奇异与近奇异协方差（N>T、共线、零方差）、
缺失值与异常值、不可行判定、预算耗尽的可行次优解、逐位确定性、种子可复现性、
并发竞争（-race）、参数扰动稳健性、再平衡成本与换手约束。

## 目录结构

```
portfolio/
  types.go      核心类型：Problem / Constraints / Result / Diagnostics
  matrix.go     稠密线性代数：Cholesky(带抖动)、Jacobi 特征值、条件数
  estimate.go   数据清洗、样本/收缩协方差、PSD 投影
  optimize.go   Dykstra 投影、均值-方差(FISTA)、风险平价(回溯线搜索)
  rebalance.go  交易成本与换手约束下的交易生成
  robust.go     扰动敏感性与稳健性诊断
  engine.go     校验、编排、线程安全缓存引擎
```
