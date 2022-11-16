CUDA_VISIBLE_DEVICES=5 python run_ppo.py --environment Humanoid-v2

CUDA_VISIBLE_DEVICES=7 python run_hebbian_sac.py --environment Walker2d-v2

CUDA_VISIBLE_DEVICES=6 python run_sac.py --environment Walker2d-v2