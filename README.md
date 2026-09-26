# portfolio-optimizer-go

本地投资组合构建与再平衡库（纯 Go，零依赖）。输入历史收益序列与约束条件，
输出满足全部硬约束的组合权重与风险指标，并提供再平衡、稳健性诊断与并发安全
的查询引擎。所有算法确定性实现：相同输入与随机种子下输出逐位一致。

## 功能总览

| 能力 | 入口 |
|---|---|
| 均值-方差 / 风险平价构建 | `Optimize` |
| 样本协方差 / 收缩估计（含缺失值、异常值、n>p 处理） | `NewSampleEstimator` / `NewShrinkageEstimator` / `EstimateWith` |
| 再平衡（交易成本、换手上限、无交易带） | `Rebalance` |
| 参数扰动稳健性诊断 | `AssessRobustness` |
| 并发安全构建与查询 | `Engine` |

## 模型假设

- 收益矩阵 `returns[t][j]`：行=时间观测，列=资产；缺失/非有限值（NaN/±Inf）
  按缺失处理，采用成对完整（pairwise-complete）估计。
- 均值-方差目标：最大化 `mu'w − (riskAversion/2)·w'Σw`。
- 风险平价目标：最小化各资产风险贡献 `RC_i = w_i(Σw)_i / w'Σw` 与其均值的
  平方偏差，使组合风险尽量均分。
- 协方差估计后强制对称化并做特征值截断（floor = 1e-10·max(λmax,1)），
  保证半正定、无 NaN；收缩目标为对角阵（保留样本方差），强度可为负值时
  自动（Ledoit-Wolf 风格，基于完整观测行估计）。
- 异常值：`WinsorizeLimit > 0` 时按 `median ± limit·MAD` 逐资产截尾。

## 约束语义（硬约束）

返回权重在 1e-8 容差内严格满足全部约束；无法满足时返回
`StatusInfeasible` 而不返回权重。

| 字段 | 语义 |
|---|---|
| `MinWeight`/`MaxWeight` | 逐资产权重上下限（缺省 0 / 1） |
| `Budget` | 权重和；零值视为缺省 1.0（全额投资） |
| `MaxLeverage` | `Σ|w_i|` 上限；0 = 不限制 |
| `GroupCaps` | 组内成员权重和 ≤ `Cap`，组可重叠 |
| `MaxTurnover` + `CurrentWeights` | `Σ|w − w_current|` 上限 |

约束校验区分两类失败：维度/参数错误（`StatusInvalidInput`，返回 error）
与逻辑不可行（`StatusInfeasible`，如 Σmin > budget、杠杆与预算矛盾）。

## 求解与终止状态

投影梯度法 + Dykstra 投影（盒、预算、杠杆 L1 球、组半空间、换手 L1 球的
凸交），确定性回溯线搜索。终止状态：

- `StatusOptimal`：在容差内收敛；
- `StatusSuboptimal`：达到迭代上限，返回当前最优**可行**解；
- `StatusTimeout`：时间预算（`TimeBudget`）耗尽或 context 取消，返回当前
  最优**可行**解；
- `StatusInfeasible`：约束互斥或投影无法找到可行点，不返回权重；
- `StatusInvalidInput`：输入非法（伴随 error）。

绝不静默返回伪最优：任何返回的权重都经过最终可行性投影与校验。

## 再平衡

`Rebalance` 给定当前/目标权重、比例交易成本、换手上限与无交易带：

1. 小于 `NoTradeBand` 的交易被抑制（避免无意义调仓）；
2. 超出 `MaxTurnover` 时交易按比例缩放（部分再平衡，`StatusSuboptimal`）；
3. 成本-收益闸门：收益（对目标的跟踪误差下降，无估计时为 L1 距离下降）
   不超过 `MinBenefit × 成本` 时放弃调仓，`Reason` 字段给出可解释说明。

## 稳健性诊断

`AssessRobustness` 对 mu 与协方差施加固定种子的高斯扰动并重新求解，报告：
逐资产权重敏感性、最大平均权重偏移、平均换手、基准集中度（Herfindahl）
与有效赌注数。平均最大偏移 ≥ 0.10 或平均换手 ≥ 0.20 判定为不稳健
（`DiagUnstable`）。扰动后的协方差重新对称化并截断特征值，保证 PSD。

## 可复现性

求解器无随机性；扰动诊断使用 `math/rand` 固定种子。相同输入 + 相同
`Seed` ⇒ 输出逐位一致（含 `float64` 位级，见 `TestBitIdenticalReproducibility`）。

## 并发与原子性

`Engine` 内部使用 `RWMutex`：构建在锁外计算，结果在单个临界区内整体写入；
`Get` 返回深拷贝，调用方无法污染缓存；context 取消的构建不写入任何结果，
已有结果不被破坏。

## 使用示例

```go
e := portfolio.NewEngine()
res, err := e.Build(ctx, portfolio.BuildRequest{
    Key:     "core",
    Returns: returns, // [][]float64, NaN = 缺失
    // nil Estimator => 自动收缩强度
    Estimation: portfolio.EstimationConfig{WinsorizeLimit: 4},
    Constraints: &portfolio.Constraints{
        MaxWeight: []float64{0.3, 0.3, 0.3, 0.3},
        GroupCaps: []portfolio.GroupCap{
            {Name: "equity", Members: []int{0, 1}, Cap: 0.6},
        },
        MaxTurnover:    0.2,
        CurrentWeights: []float64{0.25, 0.25, 0.25, 0.25},
    },
    Options: portfolio.Options{
        Objective:   portfolio.ObjectiveRiskParity,
        TimeBudget:  2 * time.Second,
        Seed:        42,
    },
})
// res.Status / res.Weights / res.Metrics / res.Diagnostics

rep, _ := portfolio.AssessRobustness(ctx, est, cons, opts,
    portfolio.RobustnessConfig{Seed: 42, NumPerturbations: 16})

rb, _ := portfolio.Rebalance(ctx, portfolio.RebalanceRequest{
    Current: current, Target: res.Weights,
    Costs: portfolio.CostModel{BuyCost: 0.001, SellCost: 0.001},
    MaxTurnover: 0.1, NoTradeBand: 0.005, MinBenefit: 1.5,
}, &est)
```

## 诊断标志

`DiagMissingData`、`DiagOutliersWinsorized`、`DiagHighDimensional`
（资产数 > 样本数）、`DiagShrinkageApplied`、`DiagEigenvalueClipped`、
`DiagNearSingular`、`DiagTurnoverCapped`、`DiagNoTradeZone`、`DiagUnstable`。

## 测试

```sh
go test ./...        # 单元 + 集成
go test -race ./...  # 并发竞争检测
```

覆盖：约束边界、奇异/近奇异协方差、缺失与异常值、n>p、换手率上限、
逐位确定性、并发安全（-race）、参数扰动稳定性。
