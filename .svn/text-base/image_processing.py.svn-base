import math
import random

def generate_disk(image_size, radius):
	'''Generates a square ASCII image of disk the given radius'''
	image_name = "disk_size" + str(image_size) + "radius_" + str(radius) + ".txt"
	image_file = open(image_name, 'w')

	center = image_size / 2
	for row in range(image_size):
		for col in range(image_size):
			if dist(row, col, center, center) < radius:
				bit = 1
			else:
				bit = 0
			image_file.write(str(bit) + " ")
		image_file.write('\n')
	return image_name

def generate_square(image_size, height, center):
	'''Generates an ASCII image of a square'''
	image_name = "square_" + str(image_size) + "height_" + str(height) + ".txt"
	image_file = open(image_name, 'w')

	top_left = center - height / 2
	bottom_right = center + height / 2
	for row in range(image_size):
		for col in range(image_size):
			if row < top_left or row > bottom_right:
				image_file.write("0 ")
			elif col < top_left or col > bottom_right:
				image_file.write("0 ")
			else:
				image_file.write("1 ")
		image_file.write('\n')

	return image_name

def generate_equalsign():
	''' Generates an ASCII image of an equal sign.'''
	image_name = "equalsign.txt"
	image_file = open(image_name, 'w')

	for row in range(100):
		for col in range(100):
			if row < 25 or row > 75 or col < 20 or col > 80:
				image_file.write("0 ")
			elif row > 45 and row < 55:
				image_file.write("0 ")
			else:
				image_file.write("1 ")
		image_file.write('\n')
	return image_name


def dist(x1, y1, x2, y2):
	''' Returns the distance between (x1, x2) and (y1, y2)'''
	return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def noisify(image_filepath, prob):
	''' Takes an image filepath and a probability (mu), and 
	flips a bit with probability mu. Used for introducing noise in an image.
	'''
	image_file = open(image_filepath, 'r')
	noise_filepath = "noise_prob" + str(prob) + "_" + image_filepath
	noise_file = open(noise_filepath, 'w')
	for line in image_file.readlines():
		line = line.strip().split()
		for bit in line:
			if random.random() < prob:
				noise_file.write(flip(bit) + " ")
			else:
				noise_file.write(bit + " ")
		noise_file.write('\n')
	return noise_filepath


def flip(bit):
	''' Flips a bit'''
	if bit == '1':
		return '0'
	elif bit == '0':
		return '1'






