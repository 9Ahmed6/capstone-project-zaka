from dataclasses import dataclass
from typing import List, Literal

@dataclass
class DiffusionConfig:
    noise_schedule: str
    diffusion_steps: int
    sigma_small: bool
    repr: Literal['joint_pos', 'joint_rot', 'joint_pos_w_scalar_rot', 'joint_pos_w_axisangle_rot']
    contact_loss: bool
    lambda_contact: float | None
    lambda_contact_predict: float | None
    lambda_rcxyz: float
    lambda_repr: float
    lambda_vel: float
    lambda_acce: float
    lambda_fc: float
    lambda_w_ig: float

@dataclass
class ModelConfig:
    latent_dim: int
    layers: int
    num_heads: int
    ff_size: int
    dropout: float
    activation: str
    cond_mode: str
    diffusion: DiffusionConfig
    contact_prediction: bool
    repr: Literal['joint_pos', 'joint_rot', 'joint_pos_w_scalar_rot', 'joint_pos_w_axisangle_rot']

@dataclass
class ActionConditionModelConfig(ModelConfig):
    cond_mask_prob: float
    num_actions: int

@dataclass
class TextConditionModelConfig(ModelConfig):
    arch: str
    text_model: str
    max_text_length: int | None
    cond_mask_prob: float
    treble_mask_prob: float



@dataclass
class DataConfig:
    ratio: float
    fixed_length: int
    max_length: int
    min_length: int
    normalize: bool
    difference: bool
    repr: Literal['joint_pos', 'joint_rot', 'joint_pos_w_scalar_rot', 'joint_pos_w_axisangle_rot']
    contact_label: bool
    data_dir: str
    use_plain: bool | None
    data_file_name: str | None

@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    shuffle: bool

@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float

@dataclass
class SampleConfig:
    guidance_param: float

@dataclass
class VisualizationConfig:
    denoising_steps: List[int]
    samples_count: int

@dataclass
class ValidationConfig:
    val_interval: int
    dataloader: DataLoaderConfig

@dataclass
class EvaluationConfig:
    dataloader: DataLoaderConfig
    eval_interval: int
    num_samples_on_train: int
    num_samples_on_val: int
    num_samples_per_condition: int

@dataclass
class TrainingConfig:
    save_dir: str
    overwrite: bool
    train_platform_type: str
    log_interval: int
    save_interval: int
    num_steps: int
    resume_checkpoint: str
    eval_during_training: bool
    eval_cfg: EvaluationConfig | None
    val_during_training: bool
    val_cfg: ValidationConfig | None
    optimizer: OptimizerConfig
    sample: SampleConfig
    dataloader: DataLoaderConfig
    viz_during_training: bool
    viz_cfg: VisualizationConfig | None


@dataclass
class Config:
    seed: int
    model: ModelConfig | ActionConditionModelConfig | TextConditionModelConfig
    data: DataConfig
    train: TrainingConfig

@dataclass
class GenerateConfig:
    model_path: str
    output_dir: str
    num_samples: int
    sample: SampleConfig
    action_name: str | None
    text_prompt: str | None
    motion_length: int