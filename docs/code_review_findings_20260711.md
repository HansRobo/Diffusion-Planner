# HDP 分支深度代码审查报告（最终合并版）

- 日期：2026-07-11
- 审查对象：`feature/hyper-diffusion-planner`，HEAD = `9197117`（含审查中途合入的 PR #220 "Replace HDP action head with temporal ego decoder"）
- 基线：`fix/all-data-model-quality-fixes`（分支 diff 约 +7.2k / -74.7k 行），另加 PR #220 增量（约 ±1.4k 行）
- 方法：max 级审查协议 —— 10 个独立查找角度（逐行扫描 / 删除行为审计 / 跨文件追踪 / 语言陷阱 / 封装正确性 / 复用 / 简化 / 效率 / 层次 / 约定）+ 全量 diff 人工精读 + 逐条机制验证。所有行号以 HEAD `9197117` 为准。
- 验证状态：`diffusion_planner/tests` 在 HEAD 上 **139 passed**；关键崩溃项已在 venv 中实际执行复现（标注"已执行验证"）。

---

## P1 — 会崩溃或静默破坏训练/部署的正确性问题（必须修）

### 1. closed-loop 验证钩子会崩掉训练主进程
- 位置：`scenario_generation/reproducer_rollout.py:222`（`_add_static_inputs`）
- 问题：仍构造 `future_len + 1`（81）步的 `sampled_trajectories`，并附带已删除的 `delay` 键；PR #220 后 temporal decoder 只接受 80 步。
- 触发：任一 trainer 带 `--closed_loop_npz_root` → 首个 save-cadence epoch 进入 `closed_loop_validate` → decoder `reshape(B,1,320)` 对 324 元素抛 RuntimeError → **rank-0 训练进程训练中途死亡**。
- 修复方向：改为 `future_len` 步、去掉 `delay`；建议同时给 decoder 入口加更友好的形状报错。

### 2. replay 的奖励评分重写版首步即崩（已执行验证）
- 位置：`scenario_generation/replay.py:136`（`compute_reward_batch`，替代已删 `rlvr.reward` 的本地重写）
- 问题：对每个张量子分数调用 `.item()`，但 `planner_metrics.aggregate.compute_subscores_batch` 返回 (N,T) 形状的 `ttc_unsafe_at_t`、`ttc_min_clearance`（`planner_metrics/aggregate.py:259-262`）。
- 触发：replay 带 `--dump_npz_dir`（此时 reward_config_path 必填）→ `_score_step` → `RuntimeError: a Tensor with N elements cannot be converted to Scalar`。metrics_log 记录路径整体不可用。
- 修复方向：区分标量/每步张量子分数（如仅对 `numel()==1` 取 item，其余取 `.tolist()` 或聚合），旧版 RewardBreakdown 只导出标量字段。

### 3. scenario_generation 所有模型前向对新 checkpoint 报废
- 位置：`scenario_generation/tensor_converter.py:704`（`to_model_tensors`）
- 问题：仍生成 `(1, P, future_len+1, 4)` 的 `sampled_trajectories`。
- 触发：`scenario_generation.simulate` 或 `replay` 加载 HDP checkpoint → 首个规划步 `shape [1,1,320] is invalid for input of size 324`。
- 修复方向：与 #1 相同，统一到 80 步；建议提取一个共享的 `build_sampled_trajectories(model_args)` 工具函数，消灭三处重复（另见 #4）。

### 4. torch2onnx --eval-npz 校验路径崩溃
- 位置：`ros_scripts/torch2onnx.py:71`（`build_inputs_from_npz`），以及 `:111`（残留 `delay` 键）
- 问题：仍构造 81 步输入；默认 dummy 路径（`onnx_export.build_dummy_inputs`）已迁移到 80 步，掩盖了该分叉。
- 触发：`torch2onnx.py <ckpt> --eval-npz <npz>` → torch 前向即抛错；即便通过，ONNX 图按 80 步 trace，ORT 侧同样拒收。发布导出的 NPZ 对比校验不可用。
- 修复方向：`OUTPUT_T + 1` → `OUTPUT_T`，删除 `delay`。

### 5. `--rl_train_scope all` 在 DDP 下必然崩溃
- 位置：`diffusion_planner/train_hdp_rl_predictor.py:713-719`；根因 `diffusion_planner/diffusion_planner/hdp_rl_utils.py:808`
- 问题：all 模式下所有参数 `requires_grad=True`，但 RL 损失前向恒设 `_skip_turn_indicator_training=True`，turn head 永不参与前向；DDP 硬编码 `find_unused_parameters=False`。
- 触发：`torchrun train_hdp_rl_predictor.py --rl_train_scope all` → 首个 backward 抛 "parameters that were not used in producing loss"。
- 修复方向：all 模式下也冻结 `decoder.turn_indicator_predictor`（与 decoder 模式一致语义），或在 all 模式下不跳过 turn head 前向，或按 scope 设 find_unused_parameters。

### 6. 弯道场景 lane 奖励被静默归零（最隐蔽，默认 RL 配置生效）
- 位置：`diffusion_planner/diffusion_planner/hdp_rl_utils.py:434`（`_hdp_lane_score` 的 kinematic_lane_change 代理）
- 问题：判据为"ego 初始坐标系横向端点位移 > lane_half_width(1.75m) 且航向变化 < 30°"。普通弯道即可同时满足：R=500m、30km/h、8s 视野 → 航向变化 ≈7.6° < 30°，横向端点偏移 ≈4.4m > 1.75m → 被判为专家变道 → 整组候选 lane 得分强制为 0。
- 影响：`rl_reward_w_lane=2.5` 是第二大权重；默认配置下 RL 在**一切非直道场景收不到车道保持信号**，无告警（仅 `lane_change_masked` 指标悄悄升高）。
- 修复方向：横向位移应相对道路/车道中心线（或最近 lane 的横向偏移变化）度量，而非初始 ego 坐标系的 y 位移；或用"相对最近车道中心线的横向偏移是否跨越半车道宽"判定。

### 7. DPO 工具在任何配置下都无法启动
- 位置：`preference_optimization/train_dpo.py:541`（及 `preference_optimization/dpo_loss.py` 的同类守卫）
- 问题：本分支新增的 `require_waypoint_diffusion` 拒绝 velocity checkpoint；而 `utils/config.py:30-35` + 新 Decoder 构造器又拒绝一切非 temporal/非 velocity checkpoint。两道门互斥，没有 checkpoint 能同时通过。
- 修复方向：要么显式删除/归档 DPO 栈（与 rlvr 同类处理），要么实现 velocity-aware 的 DPO 损失。现状是"看似保留实则死亡"的最差状态。

### 8. C++ benchmark/parity 工具无法验证新导出产物
- 位置：`cpp_tools/src/autoware_diffusion_planner_tools/src/benchmark_tool.cpp:114`
- 问题：仍按旧 `SAMPLED_TRAJECTORIES_SHAPE`（33 agents × 81 步）构造输入；本分支导出的 ONNX 全图输入为 `[B,1,80,4]` 且无 `delay`。
- 触发：benchmark_tool 跑新导出 → ORT 形状不匹配拒收。
- 修复方向：同步 C++ 侧常量与输入构造（分支文档已承认 Autoware C++ 消费端需更新，此处是同仓库内可直接修的部分）。

---

## P2 — 残留引用带（删除 rlvr/GRPO 后未清理干净）

### 9. guidance_gui 采样功能整体报废
- 位置：`guidance_gui/generate_samples.py:64-65`
- 问题：函数内 import 已整体删除的 `rlvr.closed_loop.batched_rollout` 与 `rlvr.guidance_batched`；guidance_gui 仍是根 pyproject 依赖与 workspace 成员。
- 触发：GUI 点击 Generate（`app.py:105`）→ `ModuleNotFoundError: No module named 'rlvr'`（已验证 venv 中不可解析）。

### 10. control_panel 16/18 工作流启动已删除模块
- 位置：`control_panel/workflows.py:282` 起（15 个 `rlvr.autoresearch.*` + `scenario_generation.tools.scene_branch_editor`）
- 触发：启动任一受影响工作流 → 子进程立即 `No module named 'rlvr'`（已用 importlib.find_spec 验证 16/18 不可解析）。
- 修复方向：整个 control_panel 事实报废（README 已移除其章节），建议删除或裁剪到仅存活的工作流。

### 11. replay 的 --explorer_dir 路径 import 已删模块
- 位置：`scenario_generation/replay.py:2186` 与 `:2592`（`scenario_generation.explorer_runner` 已删除）
- 触发：`replay --explorer_dir <dir>` → ModuleNotFoundError（而非文档声称的"目录缺失时响亮失败"）。SpawnConfig 的全部 explorer_* 参数一并残留。
- 修复方向：删除该功能路径与配置项，或在参数校验处直接拒绝。

### 12. exploration_policy 测试套件必然失败
- 位置：`exploration_policy/test_exploration_policy.py:287`（`from rlvr.grpo_config import GRPOConfig`）
- 触发：`pytest exploration_policy/` → ModuleNotFoundError。注意分支自述的 "139 passed" 只覆盖 `diffusion_planner/tests`。
- 修复方向：删除该测试（连同 `__main__` 运行器中的调用），或随 rlvr 一起归档 exploration_policy。

### 13. metrics_log.json 写出非标准 JSON 记号 Infinity
- 位置：`scenario_generation/replay.py:144`（`reward_breakdown_to_json_dict` 变成裸 `dict()` 拷贝）
- 问题：丢失旧实现的非有限值 → null 清洗；`rb_min_dist=inf`（无路缘数据）、`ttc_min_clearance` 默认 inf 会被 `json.dump` 写成 `Infinity`。
- 触发：严格 JSON 解析器（jq -e、JS 工具）拒绝整个文件；下游阈值筛选行为改变。
- 修复方向：恢复 `_json_safe_value` 式清洗（非有限 → null）。

### 14. 遗留 quality-fixes 启动脚本已死
- 位置：`diffusion_planner/run_all_quality_fixes_step1_base_node02.sh:67`（`--predicted_neighbor_num 320`，另有 `--decoder_depth 3`）
- 触发：执行即被新参数校验拒绝（"HDP is ego-only"）。失败是响亮的，但脚本仍被追踪且本分支还刚为它删过 `--enable_pdms_eval`。
- 修复方向：删除或明确标注废弃。

### 15. torch2onnx 死路径 KeyError（潜伏）
- 位置：`ros_scripts/torch2onnx.py:217` 与 `:225`
- 问题：`validate_split_models` 仍读取 `decoder_inputs["neighbor_agents_past"]`，但 `build_decoder_inputs` 已改为返回 `ego_current_state`。当前因 HDP velocity 恒跳过 split 导出而不可达；一旦放开守卫即 KeyError。
- 修复方向：与 #4 一并同步（改读 `ego_current_state`），或干脆删除 split-decoder 校验死路径。

---

## P3 — 训练/评测语义类问题（不崩溃但影响结果或口径）

### 16. 严格恢复后的第一个 epoch 重复 epoch-0 的数据顺序
- 位置：`diffusion_planner/diffusion_planner/train.py:1010` 与 `diffusion_planner/train_hdp_rl_predictor.py:1002`
- 问题：`train_sampler.set_epoch(epoch + 1)` 放在循环末尾；恢复到 epoch k 时，该 epoch 使用 sampler 默认 epoch=0 的洗牌顺序（与原始第 1 个 epoch 完全相同），之后才恢复正常。
- 影响：每次 Slurm requeue 断点续训都重复同一数据顺序，与本分支强调的"精确续训"语义相悖（该问题为基线遗留，但本分支把严格恢复做成卖点）。
- 修复方向：改为循环开头 `train_sampler.set_epoch(epoch)`。

### 17. 交通灯掩码命中 base∩extra 重叠样本
- 位置：`diffusion_planner/diffusion_planner/utils/dataset.py:63,80`
- 问题：`extra_train_set_mask_traffic_lights` 按路径字符串成员（`path in traffic_light_mask_paths`）判定；同一路径若同时在 base 与 extra 列表中，其 **base 侧样本也被掩码**，违反"仅对 extra 样本掩码"的文档承诺。审计文档记录 node01 base∩extra = 33,913 条重叠（当前 sbatch 用过滤后列表规避了，但 API 陷阱仍在）。
- 修复方向：按索引区间（extra 段起始下标）而非路径成员判定。

### 18. 多样本评测曾错误地从 xy 非零值推断 ego GT 有效性
- 位置：`diffusion_planner/diffusion_planner/validate_model.py:229`（`valid[no_valid] = True`）
- 最终复核：原发现不成立。ego future 没有 padding mask，数据契约固定为完整 80 帧；合法停车可以表现为全零 `(x,y,yaw)`。从 xy 是否非零推断有效性会系统性漏掉停车场景。
- 最终修复：严格按论文定义，对全部 80 个 waypoints 计算 ADE，并在第 80 帧计算 FDE。

### 19. `rl_reward_normalize=batch` 不丢弃同奖励组
- 位置：`diffusion_planner/diffusion_planner/hdp_rl_utils.py:697`
- 问题：batch 模式 `valid_sample` 只按有限性判定；奖励全相同的退化批得到 exp(0)=1 权重，变成无信号自蒸馏更新，与 group 模式/论文的丢弃规则不一致（group 模式会正确跳过 optimizer）。
- 修复方向：batch 模式同样在 std ≤ eps 时置 valid_sample=False。

### 20. `valid_group_fraction` 指标在 batch/none 模式下失真
- 位置：`diffusion_planner/diffusion_planner/hdp_rl_utils.py:924`（`valid_sample.view(num_scenes, n)[:, 0]`）
- 问题：只读候选 0 的有效性；group 模式下组内一致所以正确，batch/none 模式下 valid_sample 是逐候选的，指标错误。仅影响 W&B 健康指标，不影响损失。
- 修复方向：`valid_sample.view(num_scenes, n).any(dim=1)`（或 .float().mean()），按语义选一。

### 21. `_hdp_lane_score` 的 acos 先 clamp 后除，可产生 NaN 漏掩码
- 位置：`diffusion_planner/diffusion_planner/hdp_rl_utils.py:422-425`
- 问题：`(dot).clamp(-1,1) / (norm_prod)` —— fp32 舍入可使比值 > 1 → `acos` 返回 NaN → 两个 `heading_change <` 比较均为 False → 真正的近直线变道**不被掩码**（恰好是论文过滤所针对的场景）。
- 修复方向：先除后 clamp：`((dot)/(norm_prod)).clamp(-1,1)`。

### 22. RL trainer 恢复时 best-score 解析对 NaN 不健壮
- 位置：`diffusion_planner/train_hdp_rl_predictor.py`（恢复分支，pandas 读取 train_log.tsv 处）
- 问题：`bool(row.get("valid_full_eval", False))` 对 pandas NaN 为 True；`float(row.get("valid_epdms_total", 0.0) or 0.0)` 中 NaN 为真值直接通过。旧日志混入 NaN 时 best-score 判定可能失真。
- 修复方向：用 `pd.notna` 显式过滤。

### 23. 验证现在跑在 bf16 autocast 下（口径变化提示）
- 位置：`diffusion_planner/diffusion_planner/validate_model.py:388-390`（PR #220 新增）
- 说明：`amp_dtype=bf16` 时验证前向（含多样本评测）也 autocast；与 PR 之前的 fp32 验证日志做曲线对比时注意口径差异。属于有意的性能选择，非 bug，但需要知情。

---

## P4 — 测试覆盖与清理项

### 24. planner_metrics 奖励原语失去全部测试覆盖（战略性风险）
- 位置：`planner_metrics/subscores.py` / `geometry.py` / `aggregate.py`
- 问题：删除 rlvr 连带删光了钉住这些数值的 golden/parity 测试（test_reward_golden、test_subscores_parity、test_reward_gates、test_static_collision、test_lat_accel 等），而这些原语恰好刚成为 HDP-RL 奖励（`hdp_rl_utils` 引用 OBB clearance / road_border）与 replay 场景筛选的安全核心。当前仅剩 EPDMS 一个测试文件。
- 修复方向：为 `compute_ego_neighbor_signed_clearance`、`compute_road_border_penalty`、TTC 子分数补 golden 测试（可从已删 rlvr 测试中移植断言值）。

### 25. 死代码清理（合并为一项）
- `diffusion_planner/train_hdp_rl_predictor.py` 恢复分支：`if resume: save_path = args.save_dir else: save_path = args.save_dir` 两分支相同。
- `diffusion_planner/diffusion_planner/model/module/decoder.py` `compute_training_loss`：flow_matching / noise / score / v / 非 velocity 路径在新 Decoder 构造器强制 temporal+velocity+x_start 后全部不可达。
- `diffusion_planner/diffusion_planner/model/module/dit.py` score 模式：`x / (std.unsqueeze(-1)+1e-6)` 的广播 `[B,T,4] / [B,1]` 本身就不兼容（死路径中的潜伏崩溃）。
- `diffusion_planner/diffusion_planner/utils/data_augmentation.py` `interpolation_future_trajectory`：`num_refine < 3` 时 `P-3` 负索引静默回绕（当前默认 20，不触发）。

---

## 已验证为"不需要修"的候选（避免修复 agent 重复追查）

- ~~`sample_group` 硬编码 OUTPUT_T 在 `--future_len != 80` 时形状崩溃~~：PR #220 后 Decoder 构造器强制 `future_len == OUTPUT_T` 并响亮报错（decoder.py:296），不再可达。
- ~~`reduce_and_average_losses` 只接受 tensor~~：PR #220 已支持 python float。
- ~~delay/prefix 机制的多处一致性问题~~：机制整体移除。
- ~~epoch loss 按步存 CUDA tensor 的显存驻留~~：改为流式累加器。

## 复核通过、修复时不要"顺手改"的部分

- `sde.transform` 全部 16 组参数化恒等式、v/noise/score 目标、DiT score 缩放数学正确。
- hybrid loss 的 `_detached_integral` 前向精确、梯度窗口正确（有可执行测试钉住）。
- 奖励几何约定：`ego_shape = (wheel_base, length, width)`；neighbor/static shapes = (width, length)；`compute_ego_neighbor_signed_clearance` 的调用与之一致。
- RL 的 DDP 全局 valid-count 梯度缩放（world_size 乘子 / 全局 count 除数）正确；全 rank 无效组时所有 rank 一致跳过 optimizer。
- `ObservationNormalizer.__call__` 浅拷贝不改原 dict —— `_hdp_rl_step` 中 raw_inputs 用于奖励是安全的；设备缓存不改语义。
- `DistributedEvalSampler` 分片无重复无填充；`aggregate_valid_metrics` 的 all_reduce 调用次数在各 rank 间按 key 集对齐。
- `neighbor_future_padding_mask` 的连续前缀语义、LineEncoder 重写、encoder ego 历史取"最近 6 帧"的修正，均正确。

---

## 修复 agent 独立复核与处理结果

以下结论基于逐项源码调用链复核、官方论文 LaTeX、当前 HDP-only 模型约束和可执行回归测试，而非直接采信原报告。

| # | 复核结论 | 处理 |
|---:|---|---|
| 1 | 成立 | closed-loop 输入改为 80 步并删除 `delay`；新增形状测试。 |
| 2 | 成立 | replay 对单值 tensor 输出 scalar、逐时刻 tensor 输出 list；新增序列化测试。 |
| 3 | 成立 | `tensor_converter` 改为 80 步；非零 legacy delay 明确拒绝，零 delay 不再进入模型输入。 |
| 4 | 成立 | `torch2onnx --eval-npz` 改为 80 步且无 `delay`；已用真实 validation NPZ 验证。 |
| 5 | 成立 | `all` 和 `decoder` scope 都冻结不参与 RL loss 的 turn head；新增 scope 回归测试。 |
| 6 | 成立 | 删除 ego 初始坐标系 endpoint-y 判据，改为相对最近 lane centerline 的距离轮廓；新增弯道不误掩码、跨中心线正确掩码测试。论文未公开具体判据阈值，因此不声称阈值是官方实现。 |
| 7 | 成立 | 删除与 temporal velocity HDP 条件互斥的 DPO 训练栈及其专用 UI/LoRA 残留。 |
| 8 | 成立 | C++ benchmark 显式构造 `[1,1,OUTPUT_T,POSE_DIM]`，不再读取安装环境中的旧 joint-planner sampled shape。 |
| 9 | 成立且不仅是 import 问题 | 旧 guidance 同时依赖已删 RLVR，并被 velocity Decoder 拒绝；删除 guidance GUI 与不兼容的 classifier-guidance 实现。 |
| 10 | 成立 | control panel 的 20 个工作流中 16 个目标已不存在，且 UI 硬依赖这些 key；整体删除，不保留残缺面板。 |
| 11 | 成立 | 删除 replay 的 explorer 配置、CLI、运行分支和日志字段。旧 JSON 中未知字段仍会按现有规则忽略。 |
| 12 | 成立 | exploration policy 依赖已删除 guidance/RLVR，整体删除而非仅删除失败测试。 |
| 13 | 成立 | 递归清洗 tensor、NumPy scalar/array、list/tuple、dict 内所有 NaN/Inf 为 JSON `null`；用 `allow_nan=False` 测试。 |
| 14 | 成立 | 删除不符合 ego-only/depth-6 配置的旧 quality-fixes 启动脚本。 |
| 15 | 成立 | HDP-only 分支完整删除永远不可达的 split decoder 导出/验证路径，只保留 full/encoder/turn 图。 |
| 16 | 成立 | supervised/RL 两个循环都在每个 epoch 开始前调用 `sampler.set_epoch(epoch)`，严格 resume 不再复用 epoch-0 shuffle。 |
| 17 | 成立 | Dataset 用 extra 起始索引和 subsample stride 按 occurrence 判定，不再按路径集合命中，也不分配 958 万项来源列表。base list/NPZ 未修改。 |
| 18 | 原报告不成立，已纠正 | ego future 是固定 80 帧且无 padding mask；合法停车可为全零。multisample ADE/FDE 现按论文定义覆盖全部 80 帧，并新增静止场景测试。 |
| 19 | 成立 | “同场景候选奖励全相同则丢弃”现在独立于 normalization mode，group/batch/none 都执行；默认仍是论文的 group normalization。 |
| 20 | 成立 | `valid_group_fraction` 改为每组 `.any(dim=1)`，不再只读 candidate 0。 |
| 21 | 成立，但最佳修复不是移动 clamp | endpoint heading/acos 判据随错误的 ego-y lane-change proxy 一起删除，因此 NaN 路径不存在。 |
| 22 | 成立 | resume 日志显式处理 pandas NaN、字符串 bool、非有限 score；新增回归测试。 |
| 23 | 信息项，不是 bug | 保留 bf16 validation，并在 HDP 文档明确说明它与旧 fp32 日志不是完全相同口径。 |
| 24 | 成立 | 为 signed OBB clearance、road-border segment distance、TTC horizon/first-step 增加 golden tests。 |
| 25 | 成立 | 删除同值 save-path 分支、不可达 diffusion/flow/waypoint Decoder 与 DiT score 路径；`num_refine<3` 及 `num_refine>=horizon` 现在显式拒绝。 |

### 审计中新发现并修复

26. `MapTensorCache.get_lanes_ego` 会把无效/零填充 lane 一起选入 top-140，并在减去 ego 位姿后变成非零伪车道。现只选择 eligible lane，并在坐标变换前保存 padding mask。这个问题影响 scenario/replay encoder 输入，不影响直接读取离线 NPZ 的当前 BaseTrain。

对中途改动的额外纠正：删除了为替代已删 DPO 测试工具而临时增加、没有生产调用者的 `load_model_npz`，改为直接验证现有 `from_npz`；旧 replay JSON 的 `inference_delay` 字段仍可读取，但非零值会在配置校验阶段立即拒绝，而不是等到第一次模型前向。

### 最终验证

- 根目录测试：`345 passed, 15 skipped`。
- Ruff import lint、format check、`git diff --check`：通过。
- 当前 BaseTrain epoch-1 EMA checkpoint：成功加载到精简后的 Decoder，state dict 无缺失/多余 key。
- ONNX：full/encoder/turn checker 通过；PyTorch/ORT parity 通过；full graph 动态 batch 1/2 通过；真实 validation NPZ 输出 `[1,1,80,4]`，最大差约 `7.6e-6`。

## 第三轮独立全仓复核（同日追加）

在上述 25 项全部完成后，又对训练恢复、DDP、RL all-scope、独立验证、场景闭环和
ONNX 实机路径做了一轮独立复核，新增并修复：

27. 独立 validation 不再构造 dummy AdamW/EMA 或恢复 optimizer、scheduler、RNG；直接优先加载 EMA 权重，避免额外 GPU 状态和验证随机数被 checkpoint 覆盖。
28. `scenario_generation.load_model` 原先总是加载 live policy，现默认优先 checkpoint EMA，与训练验证和 ONNX 保持一致；真实 epoch-1 checkpoint 已逐参数确认。
29. 没有 availability tensor 的 EPDMS 子指标不再把 NaN/Inf 当作分母中的零分样本，而是从分子和分母同时排除。
30. grouped closed-loop 的两阶段 unstick 参数原先在 CLI 调用链中静默丢失，现完整传至 rollout state。
31. grouped episode 完成度原先使用整条 route 的全局最大 index，可能把完全跳过的早期 episode 标成完成；现只使用 episode 视频区间内实际访问的 index。near-miss 也不再重复计入负 clearance 的 collision steps。
32. 直接启动且没有 torchrun/Slurm 环境时，DDP fallback 原先仍保留 `ddp=True` 并在后续 collective 崩溃；现一致回退单进程。显式 `cuda:0` 在多卡 DDP 下也会按 `LOCAL_RANK` 绑定，非 DDP 的 `cuda:N` 仍被保留。
33. resume 后周期 checkpoint/closed-loop 原先相对 `init_epoch` 重新计数，可能把 epoch 10/20 漂移到 17；SFT/RL 均改为绝对 epoch cadence。
34. RL `all` scope 原先把 320x80 的 reward-only neighbor future 复制到每个候选，默认 32 generations 时可能 OOM；policy batch 现剔除 ego/neighbor supervision futures，reward 保留每场景原张量。
35. RL 启动参数新增数学有效性检查（reward 权重、normalization epsilon、时间步长和 shaping 阈值）；仍允许 occupancy source 缺失并通过 source coverage 指标显式记录，不增加用户已拒绝的严格报错模式。
36. ONNX CLI 不再在 import 时永久关闭 Flash-SDP/MHA fastpath，只在已有的可恢复导出上下文中切换；真实 NPZ 的 full/encoder/turn ORT parity 重新通过，full graph 动态 batch=2 最大差约 `1.9e-5`。
37. 训练和场景主路径的 NPZ 读取关闭 pickle，并只解压实际消费字段；正式 JT 样本已通过 SceneContext、RouteTimeline 和 ONNX 输入实读。
38. PyTorch 2.11 在分组件编译边界探测非叶子 encoder 输出时会产生上游 `.grad` 假阳性 warning；过滤范围限定在 Dynamo 内部模块。保留分组件方案，因为 H100 B64 稳态约 185 ms，而整图编译实测约 206 ms（慢约 12%）。

追加验证：根目录 `368 passed, 15 skipped`（`PYTHONWARNINGS=error`）；pre-commit 全通过；
2xH100 compiled DDP 真实 checkpoint forward/backward 后两 rank 参数 checksum 完全一致；
EMA 场景推理输出有限且形状为 `[1,1,80,4]`。
