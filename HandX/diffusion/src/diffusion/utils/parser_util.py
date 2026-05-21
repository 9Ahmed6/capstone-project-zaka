import argparse
from argparse import ArgumentParser
import json
from pathlib import Path

def parse_and_load_from_json(parser):
    '''
    从 model_path 下的 args.json 文件中加载参数，并将其覆盖到用户指定的参数中。
    '''
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)

    args = parser.parse_args() # 部分参数由用户指定
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    model_path = get_model_path_from_args()

    args_path = (Path(model_path).parent / 'args.json').as_posix()
    assert Path(args_path).exists(), f'{args_path} does not exist.'
    with open(args_path, 'r') as f:
        model_args = json.load(f) # 从模型路径下的 args.json 文件中加载参数

    for a in args_to_overwrite: # 将模型路径下的 args.json 覆盖到用户指定的参数上
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif a == 'cond_mode':
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        # else:
            # raise Warning(f"was not able to load {a}, using default value {getattr(args, a)} instead.")

    if args.cond_mask_prob == 0:
        args.guidance_param = 1.0

    return apply_rules(args)



def get_args_per_group_name(parser:ArgumentParser, args, group_name): # 根据指定的参数组名称 group_name，提取并返回该参数组内所有参数的`名称列表`
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    raise ValueError("group_name was not found")

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args() # 如果它遇到未在 dummy_parser 中定义的参数，它不会报错并退出程序，会返回一个包含（已识别参数，未识别参数）的元组
        return dummy_args.model_path

    except:
        raise ValueError("model_path argument must be specified")


def add_base_options(parser:ArgumentParser):
    group = parser.add_argument_group("base")
    group.add_argument("--seed", default=10, type=int, help='For fixing random seed')

def add_diffusion_options(parser:ArgumentParser):
    group = parser.add_argument_group("diffusion")
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help='Noise schedule type')
    group.add_argument("--diffusion_steps", default=1000, type=int, help='Number of diffusion steps (denoted T in the paper)')
    group.add_argument("--sigma_small", default=True, type=bool, help='Use smaller sigma values.')

def add_model_options(parser:ArgumentParser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc', choices=['trans_enc', 'trans_dec', 'gru'], type=str, help='Architecture types as reported in the paper') # GRU: Gated Recurrent Unit
    group.add_argument("--emb_trans_dec", default=False, type=bool, help='For trans_dec architecture only, if true, will inject condition as a class token (in addition to cross_attention).')
    group.add_argument("--layers", default=8, type=int, help='Number of layers')
    group.add_argument("--latent_dim", default=512, type=int, help='Transformer/GRU width')
    group.add_argument("--text_model", default='openai/clip-vit-base-patch32', type=str, choices=['t5-small', 't5-large', 't5-base', 'openai/clip-vit-base-patch32'], help='the text encoder model to use.')
    group.add_argument("--max_text_length", default=None, type=int, help='Maximum text length. If the text is longer, it will be truncated.')
    group.add_argument("--num_actions", default=12, type=int, help='Number of actions in the dataset. If the dataset does not have actions, this parameter will be ignored.')
    group.add_argument("--cond_mode", default='no_cond', type=str, choices=['no_cond', "text", 'action'], help='Conditioning mode.')
    group.add_argument("--cond_mask_prob", default=0.1, type=float, help='The probability of masking the condition during training. For classifier-free guidance learning.')
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help='Joint positions loss.')
    group.add_argument("--lambda_vel", default=0.0, type=float, help='Joint velocity loss.')
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_ig", default=0.0, type=float, help='interaction graph loss.')
    group.add_argument("--lambda_w_ig", default=0.0, type=float, help='weighted interaction graph loss. (ReMos)')
    # group.add_argument("--unconstrained", action='store_true', help='Model is trained unconditionally. That is, it is constrained by neither text nor action.')
    group.add_argument("--num_heads", default=4, type=int, help='Number of attention heads')
    group.add_argument("--ff_size", default=1024, type=int, help='Feed forward size')
    group.add_argument("--dropout", default=0.1, type=float, help='Dropout rate')
    group.add_argument("--activation", default='gelu', type=str, help='Activation function to use', choices=['relu', 'gelu', 'swish'])


def add_data_options(parser:ArgumentParser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='interhand2-6m_bi_hand', type=str, choices=['interhand2-6m_bi_hand', 'gigahands_bi_hand_text', 'snaphands_bi_hand_text', 'snaphands_bi_hand_action'], help='Dataset name')
    group.add_argument("--data_dir", default='data', type=str, help='Path to the dataset directory')
    group.add_argument("--repr", default='joint_pos', type=str, choices=['joint_pos', 'joint_rot'], help='Representation of the motion data.')
    group.add_argument("--difference", action='store_true', help='If True, will use difference representation of the motion data.')
    # group.add_argument("--num_actions", default=12, type=int)
    # group.add_argument("--augment", default=False, action='store_true', help='If True, will apply data augmentation.')
    group.add_argument("--fixed_length", default=0, type=int, help='If > 0, will use fixed length for the motion.')
    group.add_argument("--max_length", default=60, type=int, help='If fixed_length is set to 0, will use this value as the maximum length of the motion. If fixed_length is set to > 0, this value will be ignored.')
    group.add_argument("--min_length", default=20, type=int, help='If fixed_length is set to 0, will use this value as the minimum length of the motion. If fixed_length is set to > 0, this value will be ignored.')
    group.add_argument("--unit_length", default=4, type=int, help='If fixed_length is set to 0, will use this value as the unit length of the motion. If fixed_length is set to > 0, this value will be ignored.')
    group.add_argument("--normalize", type=bool, default=True, help='If True, will normalize the motion data.')
    group.add_argument("--random_shift", type=bool, default=True, help='If True, will apply random shift to the motion data.')

    group.add_argument("--num_workers", default=8, type=int, help='Number of workers for data loading')


def add_training_options(parser:ArgumentParser):
    group = parser.add_argument_group("training")
    group.add_argument("--save_dir", required=True, type=str, help='Path to save checkpoints and results')
    group.add_argument("--overwrite", action='store_true', help='If True, will enable to use an already existing save_dir.')
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'WandbPlatform', 'TensorboardPlatform'], type=str, help='Choose platform to log results. NoPlatform means no logging.')

    group.add_argument("--batch_size", default=64, type=int, help='Batch size during training')
    group.add_argument("--lr", default=1e-4, type=float, help='Learning rate.')
    group.add_argument("--weight_decay", default=0.0, type=float, help='Optimizer weight decay.')
    group.add_argument("--eval_batch_size", default=32, type=int, help='Batch size during evaluation loop. Do not change this unless you know what you are doing. T2m Precision calculation is based on fixed batch size 32.')
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str, help='Which split to evalutate on during training.')
    group.add_argument("--eval_during_training", action='store_true', help='If true, will run evaluation during training.')
    group.add_argument("--viz_during_training", action='store_true', help='If true, will run sample & visualization during training.')
    group.add_argument("--val_during_training", action='store_true', help='If true, will calculate validation loss during training.')
    group.add_argument("--eval_rep_times", default=3, type=int, help='Number of repetitions for evaluation loop during training.')
    group.add_argument("--eval_num_samples", default=1_000, type=int, help='If -1, will use all samples in the specified split.')
    group.add_argument("--log_interval", default=1_000, type=int, help='Log losses each N steps.')
    group.add_argument("--save_interval", default=50_000, type=int, help='Save checkpoints and run evaluation each N steps.')
    group.add_argument("--val_interval", default=10_000, type=int, help='Run validation each N steps. If val_during_training is True, will run validation each N steps.')
    group.add_argument("--num_steps", default=2400_000, type=int, help='Training will stop after the specified number of steps.')
    group.add_argument("--resume_checkpoint", default='', type=str, help='If not empty, will start from the specified checkpoint (path to model###.pt file).')
    group.add_argument("--guidance_param", default=1, type=float, help='For classifier_free sampling - specified the s parameter, as defined in the paper.')
    group.add_argument("--viz_denoising_steps", type=int, nargs='*', default=[])
    group.add_argument("--viz_samples_count", type=int, default=1, help='How many samples each process visualizes every time?')



def add_sampling_options(parser:ArgumentParser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--model_path", required=True, type=str, help='Path to model###.pt file to be sampled.')
    group.add_argument("--output_dir", default='', type=str, help='Path to results dir (auto cereated by the script). If empty, will create dir in parallel to checkpoint.')
    group.add_argument("--num_samples", default=10, type=int, help='Maximal number of prompts to sample, if loading dataset from file, this field will be ignored.')
    group.add_argument("--guidance_param", default=1, type=float, help='For classifier_free sampling - specified the s parameter, as defined in the paper.')

    # group.add_argument("--num_repetitions", default=3, type=int, help='Number of repetitions, per sample (text prompt/action)')

def add_generate_options(parser:ArgumentParser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=60, type=int, help='The length of the sampled motion [in seconds].')
    group.add_argument("--input_text", default='', type=str, help='Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.')
    group.add_argument("--text_prompt", default='', type=str, help='A text prompt to be generated. If empty, will take text prompts from dataset.')
    group.add_argument("--action_name", default='', type=str, help='An action name to be generated. If empty, will take text prompts from dataset.')

def add_evaluation_options(parser:ArgumentParser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", type=str, default=None, help='Path to model###.pt file to be sampled.')
    group.add_argument("--model_family_path", type=str, default=None, help='Path to model folder to be sampled.')
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str, help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return apply_rules(parser.parse_args())

def get_cond_mode(args):
    return args.cond_mode

def generate_args():
    parser = ArgumentParser()
    # 只有 base、sampling、generate 三个参数组是用户指定的，其他参数组的参数会从模型路径下的 args.json 文件中加载并覆盖用户指定的参数
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_json(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise ValueError("Specified text prompt or input text, but the cond_mode is not set to 'text")

    return args

def evaluation_parser():
    parser = ArgumentParser()
    # 只有 base、evaluation 两个参数组是用户指定的，其他参数组的参数会从模型路径下的 args.json 文件中加载并覆盖用户指定的参数
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_json(parser)

def apply_rules(args):
    if args.fixed_length > 0:
        del args.max_length
        del args.min_length
        del args.unit_length

    if hasattr(args, 'val_during_training') and not args.val_during_training:
        del args.val_interval

    if args.cond_mode == 'no_cond':
        del args.cond_mask_prob
        del args.max_text_length
        del args.text_model
        del args.num_actions
    elif args.cond_mode == 'text':
        del args.num_actions
    elif args.cond_mode == 'action':
        del args.max_text_length
        del args.text_model
    return args
