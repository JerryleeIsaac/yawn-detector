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

# docker: requirements.txt ## Create docker image
# 	docker build -t yawn-detector .


# results_export: ## Export results to CSV file
# 	dvc exp show -T --include-params train.model_params.classifier_type --csv --precision 4 > results.csv

# plots: ## Generate plots for current revision
# 	dvc plots show --template confusion pipeline_outs/results/predictions.csv -x actual -y predicted -o dvc_plots/confusion_matrix
# 	dvc plots show --template smooth pipeline_outs/results/precision_recall_curve.csv -x recall -y precision -o dvc_plots/precision_recall_curve