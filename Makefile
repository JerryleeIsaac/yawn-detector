.PHONY: dev format clean
.DEFAULT: help
help: ## Display this help message
	@echo "Please use \`make <target>\` where <target> is one of"
	@awk -F ':.*?## ' '/^[a-zA-Z]/ && NF==2 {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
clean: ## Remove general artifact files
	find . -name '.coverage' -delete
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '.pytest_cache' -type d | xargs rm -rf
	find . -name '__pycache__' -type d | xargs rm -rf
	find . -name '.ipynb_checkpoints' -type d | xargs rm -rf

dev: ## Install dev requirements
	poetry install
	pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit
	.venv/bin/pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

	bash build_dlib.sh
	bash -c "cd dlib/; ../.venv/bin/python3 setup.py install"
	bash -c ".venv/bin/pip3 install face-recognition==1.3.0"
	
	
run: dev ## Run yawn detector app
	streamlit run app.py


plots: ## Generate plots
	bash generate_plots.sh

results: ## Show experiment results
	dvc exp show -T