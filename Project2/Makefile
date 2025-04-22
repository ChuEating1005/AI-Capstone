train_atari:
	python src/main.py --env ALE/Pong-v5 --timesteps 2_000_000

train_classic:
	python src/main.py --env Pendulum-v1 --timesteps 2_000_000

train_box2d:
	python src/main.py --env LunarLander-v3 --timesteps 2_000_000

tensorboard:
	tensorboard --logdir results/ALE/Pong-v5/logs/ --port 6006
	