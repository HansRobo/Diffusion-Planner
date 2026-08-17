# Hydra Config System

基于 Hydra 的统一配置管理系统，用于训练和测试 Diffusion Planner。

## 目录结构

```
Diffusion-Planner/
├── configs/                       # Hydra 配置文件
│   ├── default.yaml               # 全局默认配置
│   ├── hydra/
│   │   └── default.yaml          # Hydra 全局设置
│   └── README.md
├── diffusion_planner/
│   ├── train_predictor.py        # 训练入口（支持 Hydra）
│   └── train_config.py           # TrainConfig dataclass
└── ...
```

## 使用方法

### 1. 训练

```bash
# 方式 1: 直接命令行 override（推荐用于实验）
python diffusion_planner/train_predictor.py \
    --config-name default \
    train_set_list=/path/to/train.json \
    valid_set_list=/path/to/valid.json \
    batch_size=256 \
    train_epochs=100

# 方式 2: 使用自定义配置文件
python diffusion_planner/train_predictor.py \
    --config-path /path/to/your/configs \
    --config-name your_exp \
    train_set_list=/path/to/train.json \
    valid_set_list=/path/to/valid.json
```

### 2. 重要参数

必填参数（必须显式指定）:
- `train_set_list`: 训练数据列表 JSON
- `valid_set_list`: 验证数据列表 JSON

常用 override:
- `batch_size`: 批次大小
- `train_epochs`: 训练轮数
- `learning_rate`: 学习率
- `wandb_project_name`: wandb 项目名
- `closed_loop_npz_root`: closed-loop 验证数据路径

## 数据流

```
YAML configs → Hydra DictConfig → OmegaConf.to_container() → TrainConfig(**dict)
                                                                    ↓
                                                       train.py (现有逻辑不变)
                                                                    ↓
                                                       args.json (输出不变)
```

**重要**: `args.json` 输出格式保持不变，供下游系统使用。

## 私有实验配置示例

在 Meta 仓库创建实验配置:

```yaml
# configs/my_experiment.yaml
defaults:
  - default

train_set_list: /path/to/private_train.json
valid_set_list: /path/to/private_valid.json
closed_loop_npz_root: /path/to/closed_loop_data

# Override 默认值
batch_size: 128
train_epochs: 50
wandb_project_name: my-private-project

# 模型配置
encoder_mixer_depth: 8
hidden_dim: 384
```

运行:
```bash
python DiffusionPlanner/diffusion_planner/train_predictor.py \
    --config-path /path/to/meta/configs \
    --config-name my_experiment
```

## CLI 兼容性

向后兼容，原始 argparse 模式仍然可用:

```bash
# 原始方式（无需 Hydra）
python diffusion_planner/train_predictor.py \
    --exp_name my_exp \
    --train_set_list /path/to/train.json \
    --valid_set_list /path/to/valid.json \
    --save_dir ./outputs \
    --batch_size 512
```

## 依赖

- `hydra-core>=1.3.0`
- `omegaconf>=2.3.0`
