import torch
import torchvision as tv
import streamlit as st
import face_recognition
import numpy as np
import cv2
from loguru import logger
import os


MODEL_FILE = "./pipeline_outs/models/latest-model.pth"
IMAGE_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


@st.cache(allow_output_mutation=True)
def load_model(model_file):
	model = tv.models.resnet50()
	model.fc = torch.nn.Linear(in_features=2048, out_features=2)
	model.load_state_dict(torch.load(model_file))
	model.to(DEVICE)
	model.eval()

	return model

@st.cache
def get_transforms(input_size):
	imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
	return tv.transforms.Compose([
		tv.transforms.ToTensor(),
		tv.transforms.Resize((input_size, input_size)),
		tv.transforms.Normalize(imagenet_mean, imagenet_std)
	])


def get_face_locations(images):
	logger.info("Getting face locations")
	face_locations = [
		y[0] if len(y) > 0 else None for y in face_recognition.batch_face_locations(images, batch_size=4)
	]

	return face_locations

def crop_faces(images, face_locations):
	logger.info("Cropping faces")
	face_images = []
	for image, face_locations in zip(images, face_locations):
		top, right, bottom, left = face_locations
		face_images.append(image[top:bottom, left:right])

	return face_images

def preprocess(images):
	logger.info("Preprocessing frames")
	transforms = get_transforms(IMAGE_SIZE)

	return torch.stack([ transforms(image) for image in images])

def classify(model, images):
	logger.info("Classifying images")
	softmax = torch.nn.Softmax(dim=1)
	with torch.no_grad():
		images = images.to(DEVICE)

		out = model(images)

		labels_prob = softmax(out)
		labels_prob = labels_prob.cpu().data.numpy()
		labels = np.argmax(labels_prob, axis=1)

	return labels, labels_prob[:, 1].astype(float)


def main():
	st.header("Test")

	model = load_model(MODEL_FILE)
	video_file = st.file_uploader("Upload video here", type=["avi"])

	if not video_file:
		return

	import tempfile

	tfile = tempfile.NamedTemporaryFile(delete=False)
	tfile.write(video_file.read())
	capture = cv2.VideoCapture(tfile.name)
	fps = capture.get(cv2.CAP_PROP_FPS)

	logger.info("Saving video as frames")
	frames = []
	while capture.isOpened():
		success, image = capture.read()
		if not success:
			break
		frames.append(image)

	face_locations = get_face_locations(frames)
	faces = crop_faces(frames, face_locations)
	faces = preprocess(faces)
	labels, probs = classify(model, faces)



	video = cv2.VideoWriter("yawn.mp4", fourcc, fps, (frames[0].shape[1], frames[0].shape[0])) 	
	for frame, face_location, label, prob in zip(frames, face_locations, labels, probs):
		top, right, bottom, left = face_location
		color = (0, 0, 255) if label != 1 else (0, 255, 0)
		new_frame = cv2.rectangle(frame, (left, top), (right, bottom), color)
		new_frame = cv2.putText(new_frame, "Yawn: "+ str(round(prob, 4) * 100) + "%", (left, top), 1, 1, color)
		video.write(new_frame)

	video.release()
	os.system("ffmpeg -y -i yawn.mp4 -vcodec libx264 yawn-converted.mp4")
	os.remove("yawn.mp4")

	st.header("Processed video")
	st.video("yawn-converted.mp4")


if __name__ == "__main__":
	main()