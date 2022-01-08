import math
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
import face_recognition
import os
import cv2
from loguru import logger


DATA_DIR = Path("./data")
SPLIT_DIR = DATA_DIR / "split"

def get_data_files(split_dir):
	data_files = [
		(str(split_dir / class_label / video_filename), class_label)
			for class_label in os.listdir(split_dir)
				for video_filename in os.listdir(split_dir/ class_label)
					if video_filename.endswith(".avi")
	]

	return data_files

class YawnVideoFrameDataset(torch.utils.data.IterableDataset):
	def __init__(self, data_files):
		self.data_files = data_files
		self.count = 0
	
	@staticmethod
	def frame_generator(data_files):
		for video_filename, class_label in data_files:
			capture = cv2.VideoCapture(video_filename)
			while capture.isOpened():
				success, image = capture.read()
				if not success:
					break
				face_locations = face_recognition.face_locations(image)
				if not face_locations:
					continue
				top, right, bottom, left = face_locations[0]
				yield cv2.resize(image[top:bottom, left:right], (64, 64)), class_label
				# yield cv2.resize(image, (128, 128)), class_label
			capture.release()
	
	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if not worker_info:
			return self.frame_generator(self.data_files)
		
		per_worker = int(math.ceil(len(self.data_files) / float(worker_info.num_workers)))
		worker_id = worker_info.id
		iter_start = worker_id * per_worker
		iter_end = min(iter_start + per_worker, len(self.data_files))

		return self.frame_generator(self.data_files[iter_start:iter_end])

def main():
	logger.info(f"Getting data files from {SPLIT_DIR}")
	data_files = get_data_files(SPLIT_DIR)

	logger.info("Splitting train and test data")
	train_data_files, test_data_files = train_test_split(
		data_files, test_size=0.25, 
		stratify=[class_label for _, class_label in data_files]
	)

if __name__ == "__main__":
	main()