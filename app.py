import torch
import torchvision as tv
import streamlit as st
import face_recognition
import numpy as np
import cv2
from loguru import logger
import os
from PIL import Image
from webcam import webcam


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
	for image, face_location in zip(images, face_locations):
		if not face_location:
			face_images.append(image)
			continue

		top, right, bottom, left = face_location
		face_images.append(image[top:bottom, left:right])

	return face_images


def preprocess(images):
	logger.info("Preprocessing frames")
	transforms = get_transforms(IMAGE_SIZE)

	return torch.stack([transforms(image) for image in images])


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


def batch_iterate(images, batch_size):

	start = 0
	end = min(len(images), batch_size)

	while True:
		yield images[start:end]
		start += batch_size
		end = min(end + batch_size, len(images))

		if start >= len(images):
			break


def load_image(image_file):
	img = Image.open(image_file)


	return standardize_image(img)


def standardize_image(img):
	# resize image
	img = orient_image(img)
	img = np.array(img)
	max_width = 1280
	scale = max_width / img.shape[1]
	height = int(img.shape[0] * scale)
	width = int(img.shape[1] * scale)

	img = cv2.resize(img, (width, height), cv2.INTER_AREA)
	if len(img.shape) == 3 and img.shape[2] == 4:
		img = img[:, :, :3]

	return img.copy()


def orient_image(image_file):
    """
    Orients images according to indicated meta tag
    """
    ORIENTATION_KEY = 274
    DOWN = 8
    LEFT = 6
    RIGHT = 8
    img = image_file.copy()
    exif = dict(img.getexif())
    if ORIENTATION_KEY not in exif.keys():
        return img
    if exif[ORIENTATION_KEY] == DOWN:
        img = img.rotate(180, expand=True)
    elif exif[ORIENTATION_KEY] == LEFT:
        img = img.rotate(270, expand=True)
    elif exif[ORIENTATION_KEY] == RIGHT:
        img = img.rotate(90, expand=True)
    return img


def video_mode(model):
	video_file = st.file_uploader("Upload video here", type=["avi", "mp4"])
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

	with st.spinner("Extracting faces"):
		face_locations = get_face_locations(frames)
		faces = crop_faces(frames, face_locations)
		faces = preprocess(faces)

	labels = []
	probs = []

	with st.spinner("Classifying frames"):
		for face_batch in batch_iterate(faces, 32):
			labels_, probs_ = classify(model, face_batch)
			labels.extend(labels_)
			probs.extend(probs_)

	with st.spinner("Creating video file"):
		video = cv2.VideoWriter("yawn.mp4", fourcc, fps,
		                        (frames[0].shape[1], frames[0].shape[0]))
		for frame, face_location, label, prob in zip(frames, face_locations, labels, probs):
			if not face_location:
				continue
			top, right, bottom, left = face_location
			color = (0, 0, 255) if label != 1 else (0, 255, 0)
			new_frame = cv2.rectangle(frame, (left, top), (right, bottom), color)
			new_frame = cv2.putText(
				new_frame, "Yawn: " + str(round(prob, 4) * 100) + "%", (left, top), 1, 1, color)
			video.write(new_frame)
		video.release()

	os.system("ffmpeg -y -i yawn.mp4 -vcodec libx264 yawn-converted.mp4")
	os.remove("yawn.mp4")

	st.header("Processed video")
	st.video("yawn-converted.mp4")


def image_mode(model):
	image_file = st.file_uploader(
		"Upload video here", type=["jpg", "jpeg", "png"])
	if not image_file:
		return

	image = load_image(image_file)

	with st.spinner("Extracting face"):
		[face_location] = get_face_locations([image])
		[face] = crop_faces([image], [face_location])
		faces = preprocess([face])

	with st.spinner("Classifying face"):
		[[label], [prob]] = classify(model, faces)

	with st.spinner("Creating image"):
		top, right, bottom, left = face_location
		color = (0, 0, 255) if label != 1 else (0, 255, 0)
		new_image = cv2.rectangle(image, (left, top), (right, bottom), color)
		new_image = cv2.putText(
			new_image, "Yawn: " + str(round(prob, 4) * 100) + "%", (left, top), 1, 1, color)

	st.header("Processed Image")
	st.image(new_image)


def webcam_mode(model):
	image = webcam()
	if not image:
		return
	image = standardize_image(image)

	with st.spinner("Extracting face"):
		[face_location] = get_face_locations([image])
		[face] = crop_faces([image], [face_location])
		faces = preprocess([face])

	with st.spinner("Classifying face"):
		[[label], [prob]] = classify(model, faces)

	with st.spinner("Creating image"):
		top, right, bottom, left = face_location
		color = (0, 0, 255) if label != 1 else (0, 255, 0)
		new_image = cv2.rectangle(image, (left, top), (right, bottom), color)
		new_image = cv2.putText(
			new_image, "Yawn: " + str(round(prob, 4) * 100) + "%", (left, top), 1, 1, color)

	st.header("Processed Image")
	st.image(new_image)


def main():
	st.title("Yawn Detector")

	model = load_model(MODEL_FILE)
	mode = st.selectbox("Choose Mode", options=["Video", "Image", "Webcam"])

	if mode == "Video":
		video_mode(model)
	elif mode == "Image":
		image_mode(model)
	elif mode == "Webcam":
		webcam_mode(model)


if __name__ == "__main__":
	main()
