import math
from scipy.sparse import data
from sklearn.model_selection import train_test_split
import torch
import torchvision as tv
from pathlib import Path
import face_recognition
import os
import cv2
from loguru import logger
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

LABEL_ENCODINGS = {
	"yawn": 1,
	"not-yawn": 0,
}

def get_data_files(split_dir):
	split_dir = Path(split_dir)
	data_files = [
		(str(split_dir / class_label / video_filename), class_label)
			for class_label in os.listdir(split_dir)
				for video_filename in os.listdir(split_dir/ class_label)
					if video_filename.endswith(".avi")
	]

	return data_files

def normalize_image(image):
	image = cv2.normalize(image, None, 255, cv2.NORM_MINMAX, dtype = cv2.CV_32F)
	image = (image - np.array(imagenet_mean).reshape(3, 1, 1)) / np.array(imagenet_std).reshape(3, 1, 1)

	return image

def resize_image(size):
	def resize(image):
		return cv2.resize(image, size)
	
	return resize

def hwc_to_chw(image):
	return image.reshape(3, 64, 64)
	

class YawnVideoFrameDataset(torch.utils.data.IterableDataset):
	def __init__(self, data_files, face_reco_batch_size, transforms=[]):
		super(YawnVideoFrameDataset).__init__()
		self.data_files = data_files
		self.transforms = transforms
		self.face_reco_batch_size = face_reco_batch_size
	
	@staticmethod
	def frame_generator(data_files, face_reco_batch_size, transforms=[]):
		for video_filename, class_label in data_files:
			capture = cv2.VideoCapture(video_filename)
			image_store = []
			while capture.isOpened():
				success, image = capture.read()

				if success:
					image_store.append(image)
	
				if success and len(image_store) < face_reco_batch_size:
					continue

				# face_locations = face_recognition.face_locations(image, model="cnn")
				# print(f"Image store length: {len(image_store)}")
				batch_face_locations = face_recognition.batch_face_locations(image_store)
				for image, face_locations in zip(image_store, batch_face_locations):
					if not face_locations:
						continue
					top, right, bottom, left = face_locations[0]
					image = image[top:bottom, left:right] 

					for transform in transforms:
						image = transform(image)

					yield image, LABEL_ENCODINGS[class_label]
				image_store = []

				if not success:
					break

			capture.release()
	
	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if not worker_info:
			return self.frame_generator(self.data_files, self.face_reco_batch_size, self.transforms)
		
		per_worker = int(math.ceil(len(self.data_files) / float(worker_info.num_workers)))
		worker_id = worker_info.id
		iter_start = worker_id * per_worker
		iter_end = min(iter_start + per_worker, len(self.data_files))

		return self.frame_generator(self.data_files[iter_start:iter_end], self.face_reco_batch_size, self.transforms)


def load_model(
		model_type,
		optimizer_type,
		scheduler_type,
		pretrained=True,
		learning_rate=0.001,
		device="cpu",
		**kwargs
	):
	if model_type == "MobileNetV3":
		model = tv.models.mobilenet_v3_large(pretrained=pretrained)
	
	model = model.to(device)

	criterion = torch.nn.CrossEntropyLoss()

	if optimizer_type == "SGD":
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=kwargs["momentum"])
	elif optimizer_type == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	if scheduler_type == "StepLR":
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=kwargs["step_size"], gamma=kwargs["gamma"])
	elif scheduler_type == "ReduceLROnPlateau":
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, factor=0.1, patience=kwargs["patience"]
		)

	return model, criterion, optimizer, scheduler

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
	
def main(config):
	logger.info(f"Getting data files from {config['split_dir']}")
	data_files = get_data_files(config["split_dir"])

	logger.info("Splitting train and test data")
	train_data_files, test_data_files = train_test_split(
		data_files, test_size=config["test_size"], 
		stratify=[class_label for _, class_label in data_files]
	)

	logger.info("Creating datasets")
	train_dataset = YawnVideoFrameDataset(train_data_files, config["face_reco_batch_size"], transforms=[
		resize_image((config["image_size"], config["image_size"])), hwc_to_chw, normalize_image, 
	])
	test_dataset = YawnVideoFrameDataset(test_data_files, config["face_reco_batch_size"], transforms=[
		resize_image((config["image_size"], config["image_size"])), hwc_to_chw, normalize_image, 
	])

	logger.info("Creating dataloader")
	train_dataloader = torch.utils.data.DataLoader(train_dataset,
		batch_size=config["batch_size"],
		num_workers=config["num_workers"])
	test_dataloader = torch.utils.data.DataLoader(test_dataset,
		batch_size=config["batch_size"],
		num_workers=config["num_workers"])
	
	logger.info("Getting model")
	model, criterion, optimizer, scheduler = load_model(
		config["model_type"], config["optimizer_type"], config["scheduler_type"],
		config["learning_rate"], device=DEVICE, **config["model_params"]
	)

	logger.info("Training model")


	best_score = 0
	saved_model = model
	for i in range(config["epochs"]):
		epoch_results = train_model(train_dataloader, model, criterion, optimizer, scheduler, DEVICE)

		# save best model
		# if epoch_results["f1"] > best_score:
		# 	best_score = epoch_results["f1"]



if __name__ == "__main__":
	main({
		"face_reco_batch_size": 16,
		"batch_size": 128,
		"num_workers": 0, # 0 for GPU usage
		"image_size": 64,
		"split_dir": "./data/split",
		"test_size": 0.25,
		"model_type": "MobileNetV3",
		"optimizer_type": "SGD",
		"scheduler_type": "ReduceLROnPlateau",
		"learning_rate": 0.001,
		"model_params": {
			"patience": 0.7,
			"gamma": 0.1,
			"step_size": 7,
			"momentum": 0.9
		},
		"epochs": 1
	})