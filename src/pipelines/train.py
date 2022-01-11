import time
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
import os
import json
from tqdm import tqdm
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight

def get_weighted_random_sampler(dataset):
    weights = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    return sampler


def load_model(
		model_type,
        device="cpu",
        **model_params
	):

    logger.info(f"Loading model {model_type} with {model_params}")
    if model_type == "MobileNetV3":
        model = tv.models.mobilenet_v3_large(
            pretrained=model_params["pretrained"])
        model.classifier[-1] = torch.nn.Linear(1280, 2)
    elif model_type == "ResNet50":
        model = tv.models.resnet50(pretrained=model_params["pretrained"])
        model.fc = torch.nn.Linear(in_features=2048, out_features=2)
    elif model_type == "ResNet34":
        model = tv.models.resnet34(pretrained=model_params["pretrained"])
        model.fc = torch.nn.Linear(in_features=512, out_features=2)
    elif model_type == "ResNet101":
        model = tv.models.resnet101(pretrained=model_params["pretrained"])
        model.fc = torch.nn.Linear(in_features=2048, out_features=2)
    elif model_type == "ResNet152":
        model = tv.models.resnet101(pretrained=model_params["pretrained"])
        model.fc = torch.nn.Linear(in_features=2048, out_features=2)

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
    y_actuals, y_preds = [], []

    losses = []
    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device)


        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            losses.append({"loss": loss.item()})
            optimizer.step()

            y_actuals.extend(labels.cpu().numpy().tolist())
            y_preds.extend(preds.data.cpu().numpy().tolist())

    epoch_results = get_metrics(y_actuals, y_preds)
    learning_rate = optimizer.param_groups[0]["lr"]

    scheduler.step(epoch_results["f1"])
    logger.info(f"Epoch Train results: {epoch_results}, learning rate: {learning_rate} average loss: {np.mean(losses)}")

    return epoch_results, losses

def evaluate_model(data_loader, model, device):
    model.eval()

    y_actuals, y_preds = [], []
    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probs, preds = torch.max(outputs, 1)

        y_actuals.extend(labels.cpu().numpy().tolist())
        y_preds.extend(preds.cpu().numpy().tolist())
    
    epoch_results = get_metrics(y_actuals, y_preds)

    logger.info(f"Epoch Test results: {epoch_results}")

    return epoch_results

def get_transforms(input_size):

    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return {
        "train": tv.transforms.Compose([
            tv.transforms.Resize((input_size, input_size)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
        "test": tv.transforms.Compose([
            tv.transforms.Resize((input_size, input_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
    }


def unzip_data(data_zip):
    logger.info(f"Unzipping {data_zip}")
    with ZipFile(data_zip, "r") as zip:
        zip.extractall()
    
def clean_data_files(data_dir):
    logger.info(f"Cleaning {data_dir}")
    shutil.rmtree(data_dir)

def get_dataloader(split_type, data_dir, transforms, batch_size, n_workers):
    logger.info(f"Getting data loader for {data_dir}")
    dataset = tv.datasets.ImageFolder(data_dir, transform=transforms)
    sampler=None
    if split_type == "train":
        sampler = get_weighted_random_sampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=sampler
    )

def save_results(results, results_dir, results_file):
    os.makedirs(results_dir, exist_ok=True) 
    logger.info(f"Saving results to {results_file}")

    with open(results_file, "w") as f:
        json.dump(results, f)

def save_losses(losses, results_dir, losses_file):
    os.makedirs(results_dir, exist_ok=True) 
    logger.info(f"Saving losses to {losses_file}")

    with open(losses_file, "w") as f:
        json.dump(losses, f)

def save_plots(plots, results_dir, plots_file):
    os.makedirs(results_dir, exist_ok=True) 
    logger.info(f"Saving plots to {plots_file}")

    with open(plots_file, "w") as f:
        json.dump(plots, f)

    
def save_model(weights, model_dir, model_file):
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Saving model weights to {model_file}")
    torch.save(weights, model_file)

def main(config):

    model = load_model(config["model_type"], DEVICE, **config["model_params"])
    optimizer = load_optimizer(model.parameters(), config["optimizer_type"], **config["optimizer_params"])
    criterion = load_criterion(config["model_criterion"])
    scheduler = load_scheduler(optimizer, config["scheduler_type"], **config["scheduler_params"])

    unzip_data(config["train_data_files_zip"])
    unzip_data(config["test_data_files_zip"])

    transforms = get_transforms(config["input_size"])
    train_dataloader = get_dataloader(
        "train",
        config["train_data_files_dir"],
        transforms["train"],
        config["batch_size"],
        config["n_workers"]
    )
    test_dataloader = get_dataloader(
        "test",
        config["test_data_files_dir"],
        transforms["test"],
        config["batch_size"],
        config["n_workers"]
    )

    os.makedirs(config["model_dir"], exist_ok=True)
    best_score = -1
    train_results = []
    test_results = []
    losses = []
    best_weights = model.state_dict()

    since = time.time()
    for i in range(config["epochs"]):
        logger.info(f"Epoch {i+1}: Training model")
        train_result, losses = train_model(train_dataloader, model, criterion, optimizer, scheduler, DEVICE)
        losses.extend(losses)
        train_results.append(train_result)

        logger.info(f"Epoch {i+1}: Evaluating model")
        test_result = evaluate_model(test_dataloader, model, DEVICE)
        test_results.append(test_result)

        if test_result["f1"] > best_score:
            logger.info(f"Current score {test_result['f1']} exceeds best score {best_score}")
            best_score = test_result["f1"]
            best_weights = model.state_dict() 
            save_model(best_weights, config["model_dir"], config["model_file"])

            model.load_state_dict(best_weights)
    time_elapsed = time.time() - since
    logger.info(f"Training completed in {time_elapsed / 60}m {time_elapsed % 60}s")

    model.load_state_dict(best_weights)

    logger.info("Evaluating model")
    train_result = evaluate_model(train_dataloader, model, DEVICE)
    train_result["time_elapsed"] = time_elapsed
    test_result = evaluate_model(test_dataloader, model, DEVICE)

    logger.info("Saving results")
    save_results(train_result, config["results_dir"], config["train_results_file"])
    save_results(test_result, config["results_dir"], config["test_results_file"])

    logger.info("Saving data plots")
    save_plots(train_results, config["results_dir"], config["train_plots_file"])
    save_plots(test_results, config["results_dir"], config["test_plots_file"])
    save_losses(losses, config["results_dir"], config["losses_file"])

    save_model(model.state_dict(), config["model_dir"], config["model_file"])
    
    clean_data_files(config["train_data_files_dir"])
    clean_data_files(config["test_data_files_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains model")
    parser.add_argument("--params", help="Params file", required=True)
    args = parser.parse_args()

    config = None
    with open(args.params, "r") as f:
        config = yaml.safe_load(f)["train"]

    main(config)
