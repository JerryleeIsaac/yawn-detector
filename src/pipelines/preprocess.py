import argparse
import yaml
from loguru import logger
import cv2
import face_recognition
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
from zipfile import ZipFile
import shutil


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

def frame_generator(data_files, face_reco_batch_size):
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

				yield image, LABEL_ENCODINGS[class_label]
			image_store = []

			if not success:
				break
		capture.release()

def main(config):
	data_files = get_data_files(config["split_dir"])

	image_files = []
	labels = []
	for i, (image, label) in enumerate(frame_generator(data_files, config["face_reco_batch_size"])):
		label_dir = f"{config['data_files_dir']}/{label}"
		if not os.path.isdir(label_dir):
			logger.info(f"Creating label dir {label_dir}")
			os.makedirs(label_dir, exist_ok=True)

		filename = f"{config['data_files_dir']}/{label}/{i}.jpg"
		logger.info(f"Writing image file {filename}")
		cv2.imwrite(filename, image)
		image_files.append(filename)
		labels.append(label)
		 
	logger.info("Splitting train and test data")
	train_image_files, test_image_files = train_test_split(
		image_files, test_size=config["test_size"], 
		stratify=labels
	)

	train_data_files = []
	for filename in train_image_files:
		new_filename = filename.replace(config["data_files_dir"], config["train_data_files_dir"])
		logger.info(f"Renaming {filename} to {new_filename}")
		os.renames(filename, new_filename)
		train_data_files.append(new_filename)

	with ZipFile(config["train_data_files_zip"], "w") as zip:
		for filename in train_data_files:
			zip.write(filename)

	shutil.rmtree(config["train_data_files_dir"])

	test_data_files = []
	for filename in test_image_files:
		new_filename = filename.replace(config["data_files_dir"], config["test_data_files_dir"])
		logger.info(f"Renaming {filename} to {new_filename}")
		os.renames(filename, new_filename)
		test_data_files.append(new_filename)

	with ZipFile(config["test_data_files_zip"], "w") as zip:
		for filename in test_data_files:
			zip.write(filename)

	shutil.rmtree(config["test_data_files_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video streams into face image files")
    parser.add_argument("--params", help="Params file", required=True)
    args = parser.parse_args()

    config = None
    with open(args.params, "r") as f:
        config = yaml.safe_load(f)["preprocess"]

    main(config)
