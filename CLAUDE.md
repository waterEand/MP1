# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MP1 is a robotic manipulation policy learning framework implementing the **MeanFlow** algorithm — a flow-matching-based generative model that achieves 1-step inference (1-NFE) for real-time robot control. It uses point cloud observations, PointNet encoders, and a 1D temporal UNet for action prediction.

## Commands

### Installation
See `install.md` for full setup. Key requirements: Python 3.8, PyTorch 2.2.1, CUDA 11.8, MuJoCo 2.1.0.

```bash
# Install the main package
cd MP1 && pip install -e .

# Install third-party environments
cd third_party/gym-0.21.0 && pip install -e . && cd ../..
cd third_party/Metaworld && pip install -e . && cd ../..
cd third_party/mujoco-py-2.1.2.14 && pip install -e . && cd ../..
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
```

### Training
```bash
# Recommended: use auto_run wrapper
bash auto_run.bash <GPU_ID> <CONFIG> [TASK] [SEED]
# Example:
bash auto_run.bash 0 mp1 metaworld_reach 42

# Direct Hydra invocation (from MP1/ directory)
python train.py --config-name=mp1.yaml task=metaworld_reach training.device="cuda:0" training.seed=42
```

### Data Generation
```bash
bash scripts/gen_demonstration_metaworld.sh <task-name>   # e.g., drawer-close
bash scripts/gen_demonstration_adroit.sh <task-name>       # e.g., door
```

### Evaluation
```bash
bash scripts/eval_policy.sh <CONFIG> <TASK> <GPU_ID>
# Or directly:
cd MP1 && python eval.py --config-name=mp1.yaml task=metaworld_reach training.device="cuda:0"
```

## Architecture

### Configuration System
All hyperparameters are managed via **Hydra + YAML** in `MP1/mp1/config/`. Two main config variants:
- `mp1.yaml` — full MP1 with dispersive loss (`policy/meanpolicy_dis.py`)
- `mp.yaml` — MeanFlow without dispersive loss (`policy/meanpolicy.py`)

Task configs live in `mp1/config/task/` and specify observation shape, action dim, dataset path, and eval environment.

### Data Pipeline
Expert demonstrations are stored in **zarr format** under `MP1/data/`. The dataset classes (`mp1/dataset/`) load these, apply sequence sampling with padding, and compute normalization statistics (`LinearNormalizer`). All observations and actions are normalized before training.

### Core Inference Pipeline
```
Observation (point cloud 512×3 + agent_pos 9D)
  → MP1Encoder (PointNet + MLP) → feature vector
  → ConditionalUnet1D_MeanFlow (with FiLM conditioning + cross-attn)
  → velocity prediction → 1-step ODE integration
  → action trajectory (horizon=4)
```

### Policy Module (`mp1/policy/`)
- `meanpolicy_dis.py` — **primary policy** used in `mp1.yaml`; adds dispersive loss on embeddings to improve generalization
- `meanpolicy.py` — base MeanFlow policy without dispersive loss
- Both use the same encoder + UNet architecture; they differ only in the training loss

### Model Module (`mp1/model/`)
- `vision/pointnet_extractor.py` — `MP1Encoder`: encodes point clouds + low-dim state into a single feature vector
- `mean/conditional_unet1d_meanflow.py` — temporal UNet backbone; FiLM layers take the time/noise embedding, cross-attention takes the observation features
- `mean/ema_model.py` — EMA wrapper used during training; the EMA copy is what gets evaluated

### Training Loop (`MP1/train.py`)
`TrainMP1Workspace.run()` manages the full lifecycle:
1. Initialize policy, optimizer, LR scheduler, EMA
2. Per epoch: train batches → update EMA → optionally run env rollouts → checkpoint
3. Top-K checkpoints tracked by success rate via `TopKCheckpointManager`
4. Logging via **SwanLab** (primary) and optionally WandB

### SDE / Flow Matching (`mp1/sde_lib.py`, `mp1/losses.py`)
- `ConsistencyFM`: defines the flow ODE and provides `euler_ode()` for single-step inference
- `get_consistency_flow_matching_loss_fn()`: computes the velocity-matching loss during training
- Key hyperparameters: `flow_ratio=0.50`, `cfg_ratio=0.10`, `cfg_scale=2.0`, time sampled from lognormal

### Environment Runners (`mp1/env_runner/`)
Used only at eval time. `MetaworldRunner` / `AdroitRunner` run N episodes, compute success rate, and optionally record videos to `MP1/save_videos/`.

## Supported Tasks
- **Meta-World**: reach, push, pick-place, drawer-open, drawer-close, button-press, etc.
- **Adroit**: door, hammer, pen, relocate

## Key File Locations
| Purpose | Path |
|---|---|
| Main training entry | `MP1/train.py` |
| Primary policy | `MP1/mp1/policy/meanpolicy_dis.py` |
| PointNet encoder | `MP1/mp1/model/vision/pointnet_extractor.py` |
| UNet backbone | `MP1/mp1/model/mean/conditional_unet1d_meanflow.py` |
| Flow loss | `MP1/mp1/losses.py` |
| SDE / ODE sampler | `MP1/mp1/sde_lib.py` |
| Main config | `MP1/mp1/config/mp1.yaml` |
| Task configs | `MP1/mp1/config/task/*.yaml` |
| Checkpoint utils | `MP1/mp1/common/checkpoint_util.py` |
