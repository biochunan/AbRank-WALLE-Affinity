SHELL:=/usr/bin/zsh
.PHONY: setup-gpu-env setup-cpu-env

# NOTE: only run this inside devcontainer
devcontainer-install-tools:
	@sudo apt update && sudo apt install unzip htop nvtop tmux parallel bc -y

conda-create-env-waffle-gpu:
	@source ${HOME}/.zshrc && conda init zsh > /dev/null && \
	conda env update -f conda-env/waffle-gpu.yaml && \
	conda activate waffle && \
	pip install -Ue .


# --------------------------------------------------------
# waffle - train abrank ranking model
# --------------------------------------------------------
train-abrank-ranking-dev:
	@python waffle/train-abrank-ranking.py \
		trainer.max_epochs=1 \
		trainer.limit_train_batches=10 \
		trainer.limit_val_batches=10 \
		trainer.limit_test_batches=10 \
		logger.wandb.tags="['dev']" \
		model_init.method="lazy_init" \
		callbacks.model_checkpoint.save_top_k=1 \
		callbacks.model_checkpoint.save_last=True \
		dataset.datamodule.batch_size=32 \
		dataset.datamodule.num_workers=4

# fast overfitting
train-abrank-ranking-fast-overfit:
	@python waffle/train-abrank-ranking.py \
		trainer.max_epochs=100 \
		+trainer.overfit_batches=1 \
		logger.wandb.tags="['fast-overfit']" \
		model_init.method="lazy_init" \
		callbacks.model_checkpoint.save_top_k=0 \
		callbacks.model_checkpoint.save_last=False
# save_top_k=0: don't save any checkpoints
# save_top_k=-1: save all checkpoints
# save_top_k=k: save top k checkpoints

# NOTE: source ./.env before running this command
train-abrank-ranking-balanced-train-swapped:
	@python waffle/train-abrank-ranking.py \
		trainer.max_epochs=100 \
		logger.wandb.tags="['train', 'ranking', 'balanced-train-swapped']" \
		optimizer.lr=1e-5 \
		dataset.datamodule.train_split_path="${DATA_PATH}/AbRank/processed/splits/Split_AF3/balanced-train-swapped.csv" \
		loss_func.margin_ranking_loss.margin=0.1 \
		callbacks.model_checkpoint.save_top_k=1 \
		callbacks.model_checkpoint.save_last=True \
		dataset.datamodule.batch_size=32 \
		dataset.datamodule.num_workers=4 \
		trainer.accumulate_grad_batches=8

# NOTE: source ./.env before running this command
train-abrank-ranking-hard-ab-train-swapped:
	@python waffle/train-abrank-ranking.py \
		trainer.max_epochs=100 \
		optimizer.lr=1e-5 \
		logger.wandb.tags="['train', 'ranking', 'hard-ab-train-swapped']" \
		dataset.datamodule.train_split_path="${DATA_PATH}/AbRank/processed/splits/Split_AF3/hard-ab-train-swapped.csv" \
		loss_func.margin_ranking_loss.margin=0.1 \
		callbacks.model_checkpoint.save_top_k=1 \
		callbacks.model_checkpoint.save_last=True \
		dataset.datamodule.batch_size=32 \
		dataset.datamodule.num_workers=4 \
		trainer.accumulate_grad_batches=8

# this set only contains 12k combinations
# NOTE: source ./.env before running this command
train-abrank-ranking-hard-ag-train-swapped:
	@python waffle/train-abrank-ranking.py \
		trainer.max_epochs=100 \
		logger.wandb.tags="['train', 'ranking', 'hard-ag-train-swapped']" \
		dataset.datamodule.train_split_path="${DATA_PATH}/AbRank/processed/splits/Split_AF3/hard-ag-train-swapped.csv" \
		loss_func.margin_ranking_loss.margin=0.1 \
		callbacks.model_checkpoint.save_top_k=1 \
		callbacks.model_checkpoint.save_last=True \
		dataset.datamodule.batch_size=32 \
		dataset.datamodule.num_workers=4 \
		trainer.accumulate_grad_batches=4


# --------------------------------------------------------
# waffle - train AbRank Regression model
# --------------------------------------------------------
train-abrank-regression-dev:
	@python waffle/train-abrank-regression.py \
		trainer.max_epochs=100 \
		logger.wandb.tags="['dev', 'regression']" \
		model_init.method="lazy_init" \
		callbacks.model_checkpoint.save_top_k=1 \
		callbacks.model_checkpoint.save_last=True \
		dataset.datamodule.batch_size=32 \
		dataset.datamodule.num_workers=4

train-abrank-regression-fast-overfit:
	@python waffle/train-abrank-regression.py \
		trainer.max_epochs=100 \
		+trainer.overfit_batches=1 \
		logger.wandb.tags="['fast-overfit', 'regression']" \
		model_init.method="lazy_init" \
		callbacks.model_checkpoint.save_top_k=0 \
		callbacks.model_checkpoint.save_last=False

# NOTE: source ./.env before running this command
train-abrank-regression-balanced-train-swapped:
	@python waffle/train-abrank-regression.py \
		trainer.max_epochs=100 \
		logger.wandb.tags="['train', 'regression', 'balanced-train']" \
		optimizer.lr=1e-5 \
		dataset.datamodule.train_split_path="${DATA_PATH}/AbRank/processed/splits-regression/Split_AF3/balanced-train-regression.csv" \
		callbacks.model_checkpoint.save_top_k=1 \
		callbacks.model_checkpoint.save_last=True \
		dataset.datamodule.batch_size=32 \
		dataset.datamodule.num_workers=4 \
		trainer.accumulate_grad_batches=8

# NOTE: source ./.env before running this command
train-abrank-regression-hard-ab-train-swapped:
	@python waffle/train-abrank-regression.py \
		trainer.max_epochs=100 \
		logger.wandb.tags="['train', 'regression', 'hard-ab-train']" \
		optimizer.lr=1e-5 \
		dataset.datamodule.train_split_path="${DATA_PATH}/AbRank/processed/splits-regression/Split_AF3/hard-ab-train-regression.csv" \
		callbacks.model_checkpoint.save_top_k=1 \
		callbacks.model_checkpoint.save_last=True \
		dataset.datamodule.batch_size=32 \
		dataset.datamodule.num_workers=4 \
		trainer.accumulate_grad_batches=8

# NOTE: source ./.env before running this command
train-abrank-regression-hard-ag-train-swapped:
	@python waffle/train-abrank-regression.py \
		trainer.max_epochs=100 \
		logger.wandb.tags="['train', 'regression', 'hard-ag-train']" \
		optimizer.lr=1e-5 \
		dataset.datamodule.train_split_path="${DATA_PATH}/AbRank/processed/splits-regression/Split_AF3/hard-ag-train-regression.csv" \
		callbacks.model_checkpoint.save_top_k=1 \
		callbacks.model_checkpoint.save_last=True \
		dataset.datamodule.batch_size=32 \
		dataset.datamodule.num_workers=4 \
		trainer.accumulate_grad_batches=16
