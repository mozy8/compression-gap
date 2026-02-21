# OAT: Ordered Action Tokenization

[[Paper]](https://arxiv.org/abs/2602.04215) | [[Webpage]](https://ordered-action-tokenization.github.io/)

[Chaoqi Liu](https://chaoqi-liu.com)<sup>1</sup>, 
[Xiaoshen Han](https://xshenhan.github.io/)<sup>1</sup>, 
[Jiawei Gao](https://gao-jiawei.com/)<sup>1</sup>, 
[Yue Zhao](https://zhaoyue-zephyrus.github.io/)<sup>2</sup>, 
[Haonan Chen](https://haonan16.github.io/)<sup>1</sup>, 
[Yilun Du](https://yilundu.github.io/)<sup>1</sup>

<sup>1</sup>Harvard University
<sup>2</sup>Stanford University


## Quick start
1. Make sure `third_party/LIBERO` is fetched as a submodule:

   ```bash
   # fresh clone
   git clone --recurse-submodules git@github.com:Chaoqi-LIU/oat.git
   cd oat

   # if you already cloned without --recurse-submodules
   git submodule update --init --recursive

   # after git pull, refresh submodule pointers
   git submodule update --init --recursive
   ```

### option 1: uv
2. Install `uv` if you do not already have it. Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. Initialize the project and install all dependencies and local editable sources:

   ```bash
   uv sync
   uv pip install -e .
   ```

### option 2: (micro)conda/mamba
2. Install `micromamba` if you do not already have it. Follow [micromamba installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

3. Initialize the project and install all dependencies and local editable sources:
   ```bash
   micromamba env create -f conda_env.yaml
   ```

__NOTE__: We encountered issues running `uv` on our slurm cluster, so we also provide `conda`/`mamba` as an alternative. The example commands below use `uv`. If you have trouble setting it up or observe significant performance degradation, please let me know.

## Preparing LIBERO datasets

We provide a prebuilt `libero10` dataset on Hugging Face: [chaoqi-liu/libero10_N500.zarr](https://huggingface.co/datasets/chaoqi-liu/libero10_N500.zarr/resolve/main/libero10_N500.zarr.zip?download=true). Alternatively, follow the instructions below to build the dataset locally.

1. Download the LIBERO releases (e.g., `libero_spatial`, `libero_object`, `libero_goal`, `libero_100`) into `data/libero/hdf5_datasets/`. (`libero10` is contained in `libero100`)

    ```bash
    uv run third_party/LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_[spatial/object/goal/100]
    ```

2. Convert each HDF5 dump into the repo's zarr format:

   ```bash
   uv run scripts/convert_libero_dataset.py --root_dir data/libero --hdf5_dir_name hdf5_datasets
   ```

   The script converts every `*.hdf5` file it finds, saves `task_N{episodes}.zarr` under `data/libero/`, and prompts before overwriting existing exports. Use `-n/--num_sample_demo` to limit how many demos per task if needed.

3. Compose a `libero10` multitask zarr:

   ```bash
   uv run scripts/compose_libero_multitask_dataset.py --multitask_name libero10 --root_dir data/libero
   ```

   This merges `*.zarr` datasets related to `libero10` using `scripts/merge_data.py`, shuffles the episodes, and writes `data/libero/libero10_N{total}.zarr`.

   For SigLIP experiments on `libero90`, compose the `libero90` multitask zarr as well:

   ```bash
   uv run scripts/compose_libero_multitask_dataset.py --multitask_name libero90 --root_dir data/libero
   ```

## Train OAT Tokenizer

After you have `data/libero/libero10_N{n}.zarr` ready, train the action tokenizer that OAT policies consume:

```bash
HYDRA_FULL_ERROR=1 uv run accelerate launch \
    --num_machines [num_node] \
    --multi_gpu \
    --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10
```

## Train OAT Policy

Once the tokenizer checkpoint exists, train the policy that predicts action tokens and decodes them back into actions:

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_machines [num_node] \
    --multi_gpu \
    --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=libero/libero10 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=[path/to/oattok.ckpt]
```
set `lazy_eval=false` would evaluate policy during training every `training.rollout_every` epochs.

For SigLIP-based policy training (default task: `libero90` in `train_oatpolicy_siglip`):

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_machines [num_node] \
    --multi_gpu \
    --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy_siglip \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=[path/to/oattok.ckpt]
```

The SigLIP config freezes the vision encoder (`optimizer.obs_enc_lr=0.0`) and uses `batch_size=64` by default.

You can also override the Robomimic vision encoder capacity from CLI (same keys work for `train_oatpolicy`, `train_binpolicy`, `train_fastpolicy`, and `train_diffpolicy`):

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
    --num_machines [num_node] \
    --multi_gpu \
    --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=libero/libero10 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=[path/to/oattok.ckpt] \
    policy.obs_encoder.vision_encoder.feature_dimension=256 \
    policy.obs_encoder.vision_encoder.num_kp=128 \
    policy.embed_dim=256 \
    policy.n_layers=4 \
    policy.n_heads=4
```

## Evaluate OAT Policy on LIBERO

Evaluate a trained checkpoint using `scripts/eval_policy_sim.py`:

```bash
uv run scripts/eval_policy_sim.py \
  --checkpoint [path/to/oatpolicy.ckpt] \
  --output_dir output/eval/libero10 \
  --num_exp 5  # run 5 times, so we can get stderr
```

The script instantiates the same LIBERO runner and dataset from `oat.config.task.policy.libero.libero10` and dumps per-checkpoint statistics plus optional videos to `output/eval/libero10`.

## Further reading

Also checkout [sim_env](https://github.com/Chaoqi-LIU/sim_env), which provides a set of simulation benchmarks. 

## Citation

If you like this work, please cite:

```bibtex
@misc{liu2026oatorderedactiontokenization,
      title={OAT: Ordered Action Tokenization}, 
      author={Chaoqi Liu and Xiaoshen Han and Jiawei Gao and Yue Zhao and Haonan Chen and Yilun Du},
      year={2026},
      eprint={2602.04215},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.04215}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
