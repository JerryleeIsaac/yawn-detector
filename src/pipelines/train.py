
from os import sched_getscheduler
from sklearn import model_selection
import torch
import torchvision as tv
import argparse
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from loguru import logger
from zipfile import ZipFile
import shutil

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(
		model_type,
        device="cpu",
        **model_params
	):

    logger.info(f"Loading model {model_type} with {model_params}")
    if model_type == "MobileNetV3":
        model = tv.models.mobilenet_v3_large(pretrained=model_params["pretrained"])
	
    model = model.to(device)

    return model

def load_criterion(model_criterion):
    logger.info(f"Loading criterion {model_criterion}")
    if model_criterion == "CrossEntropy":
        return torch.nn.CrossEntropyLoss()

def load_optimizer(trainable_params, optimizer_type, **optimizer_params):
    logger.info(f"Loading optimizer {optimizer_type} with {optimizer_params}")

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=optimizer_params["learning_rate"],
            momentum=optimizer_params["momentum"]
        )
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=optimizer_params["learning_rate"]
        )

    return optimizer

def load_scheduler(optimizer, scheduler_type, **scheduler_params):
    logger.info(f"Loading scheduler {scheduler_type} with {scheduler_params}")

    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params["step_size"],
            gamma=scheduler_params["gamma"]
        )
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=scheduler_params["factor"],
            patience=scheduler_params["patience"]
        )
    
    return scheduler



def get_metrics(y_actuals, y_preds):
	return {
		"accuracy": accuracy_score(y_actuals, y_preds),
		"precision": precision_score(y_actuals, y_preds),
		"recall": recall_score(y_actuals, y_preds),
		"f1": f1_score(y_actuals, y_preds),
	}


def train_model(data_loader, model, criterion, optimizer, scheduler, device):
	model.train()

	y_actuals, y_preds = [], []

	for i, (inputs, labels) in enumerate(data_loader):
		inputs = inputs.to(device, dtype=torch.float)
		labels = labels.to(device)

		logger.info(f"Batch: {i}, Images: {len(inputs)} ")

		optimizer.zero_grad()

		with torch.set_grad_enabled(True):
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			y_actuals.extend(preds.cpu().numpy().tolist())
			y_preds.extend(labels.data.cpu().numpy().tolist())
		logger.debug("Batch run done")

	epoch_results = get_metrics(y_actuals, y_preds)
	learning_rate = optimizer.param_groups[0]["lr"]

	scheduler.step(epoch_results["f1"])
	logger.info(f"Epoch results: {epoch_results}, learning rate: {learning_rate}")

	return epoch_results

def get_transforms(input_size):

    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return {
        "train": tv.transforms.Compose([
            tv.transforms.Resize(input_size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
        "test": tv.transforms.Compose([
            tv.transforms.Resize(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }


def unzip_data(data_zip):
    logger.info(f"Unzipping {data_zip}")
    with ZipFile(data_zip, "r") as zip:
        zip.extractall()
    
def main(config):

    model = load_model(config["model_type"], DEVICE, **config["model_params"])
    optimizer = load_optimizer(model.parameters(), config["optimizer_type"], **config["optimizer_params"])
    criterion = load_criterion(config["model_criterion"])
    scheduler = load_scheduler(optimizer, config["scheduler_type"], **config["scheduler_params"])

    # unzip_data(config["train_data_files_zip"])
    # unzip_data(config["test_data_files_zip"])

    # shutil.rmtree(config["train_data_files_dir"])
    # shutil.rmtree(config["test_data_files_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains model")
    parser.add_argument("--params", help="Params file", required=True)
    args = parser.parse_args()

    config = None
    with open(args.params, "r") as f:
        config = yaml.safe_load(f)["train"]

    main(config)
