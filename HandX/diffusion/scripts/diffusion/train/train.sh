torchrun --nproc_per_node=2 --master_port=25535 -m src.diffusion.train.train \
    train.save_dir=./exps/data100 \
    data.data_dir=../data/handx \