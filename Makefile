IMAGE_NAME := akkadeeemikk/mats
CONTAINER_NAME := research

build_mats:
	docker build -f docker/Dockerfile -t $(IMAGE_NAME) .

stop:
	docker stop $(CONTAINER_NAME)

jupyter:
	jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=mats

run_docker:
	docker run -d -it --rm \
		--ipc=host \
		--network=host \
		--gpus=all \
		-v ./:/workspace/ \
		-v ./.cache/huggingface:/root/.cache/huggingface \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) bash

create_env:
	mv .env_template .env

train_sft_test:
	python vector_sft_train.py --config configs/vector_sft/test.yaml

dump_data:
	aws s3 sync ./ s3://ai-heap/mats/vector_sft/ \
		--exclude "**/__pycache__/*" --exclude "*.ipynb_checkpoints*" \
		--exclude "**/.DS_Store" --exclude "*.git*" --exclude "**/.Trash/*" \
		--exclude "*.idea*" --exclude "*vscode*" --exclude "**/.git/*" \
		--exclude "hub/*" --exclude "models/*" --exclude "**/dataset/**/*" --exclude "*SFT*/**" --exclude "*.cache/**"


craken:
	python vector_sft_train.py configs/vector_sft/llama3_2_3b_custom.yaml