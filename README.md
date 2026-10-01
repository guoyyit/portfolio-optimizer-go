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

1. **最小调仓阈值（抑制）**：某资产目标调仓量（或经重配平/换手缩放后的
   有效调仓量）满足 `|trade| < NoTradeBand` 时被抑制，该资产保持当前权重。
2. **重配平（预算闭合）**：抑制会打破“买入总额 = 卖出总额”。算法对较大
   的一侧（全部买单或全部卖单）按同一系数等比缩小，使两侧重新相等，从而
   保证 `Σnew = 1`。缩放只减小调仓量、永不改变方向，因此不会越过目标。
   重配平或换手缩放可能使更多资产跌破阈值，这些资产被再次抑制并重新配平
   （每轮至少剔除一个资产，最多 n 轮收敛）。
3. **换手上限**：重配平后的方案若 `Σ|trade| > MaxTurnover`，全部交易按
   同一系数缩放到上限；缩放保留闭合关系。缩放后若产生低于阈值的交易，
   回到第 2 步级联处理，因此不会借缩放绕过阈值，也不会借重配平绕过换手上限。
4. **成本-收益闸门**：收益（对目标的跟踪误差下降；无估计时为 L1 距离下降）
   不大于 0，或不超过 `MinBenefit × 成本` 时放弃调仓。

### 返回状态与出口语义

| 状态 | 触发条件 | 权重/交易 |
|---|---|---|
| `StatusOptimal` | 已在目标点；或无需抑制/未触发换手上限即可执行 | `new = current + trades`，和为 1 |
| `StatusSuboptimal` | 部分执行：有资产被阈值抑制并重配平，或换手上限绑定 | 新权重和为 1；被抑制资产保持当前权重 |
| `StatusAbandoned` | 全部目标调仓都低于阈值；重配平后某一侧完全消失；换手缩放后无任何满足阈值的交易；成本-收益闸门放弃 | 零交易，`new = current`，换手与成本均为 0 |
| `StatusInvalidInput` | 非法输入（见下），伴随可区分 error | 不返回权重 |

每次结果的 `Reason` 说明：哪些资产因阈值未调仓（逐资产诊断）、剩余买单/
卖单以何系数重配平、换手是否绑定及缩放比例、实际换手
`Turnover = Σ|trade|` 与成本 `Cost = Σ buy·BuyCost + Σ sell·SellCost` 的
计算口径、以及闸门放弃的收益/成本对比。相关诊断标志：
`DiagNoTradeZone`（逐资产抑制 / 闸门）、`DiagTradesReclosed`（重配平）、
`DiagTurnoverCapped`（换手上限绑定）、`DiagRebalanceAbandoned`（完全放弃）。

### 输入校验与最终结果校验

输入被明确拒绝（`StatusInvalidInput` + 可区分 error）的情形：当前与目标
长度不一致或为空；权重或估计含 NaN/±Inf；当前或目标权重和在
`1e-9·n` 容差内不等于 1；`NoTradeBand`、`MaxTurnover`、`MinBenefit` 或
买卖成本率为负/非有限；估计维度不匹配。

**所有携带权重的返回路径**（含完全放弃）都经过统一的最终校验：权重和为 1、
长度与有限性、`new = current + trades`、每笔交易方向朝向目标、非零交易不
低于阈值、换手不超上限、换手/成本字段与交易一致、放弃出口严格等于当前
权重。任一条件不满足时返回 `StatusInfeasible` 与哨兵错误
`ErrInvalidRebalanceResult`（附 `DiagInvalidResult` 诊断），绝不静默返回
权重和不为 1 或违反约束的结果。

`Rebalance` 是纯函数：不修改请求（内部使用副本）、无共享可变状态，可安全
并发调用；相同输入逐位一致。

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
`DiagNearSingular`、`DiagTurnoverCapped`、`DiagNoTradeZone`、
`DiagTradesReclosed`、`DiagRebalanceAbandoned`、`DiagInvalidResult`、
`DiagUnstable`。

## 测试

```sh
go test ./...        # 单元 + 集成
go test -race ./...  # 并发竞争检测
```

覆盖：约束边界、奇异/近奇异协方差、缺失与异常值、n>p、换手率上限、
再平衡的单边/多边（同向与反向）阈值抑制、重配平闭合、阈值与换手上限
叠加（部分执行与完全放弃）、成本-收益闸门放弃、所有返回路径的权重和与
约束校验、非法输入的可区分拒绝、逐位确定性、并发安全（-race）、
参数扰动稳定性。
