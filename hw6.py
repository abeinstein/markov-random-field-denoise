import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import os
import image_processing

THETA = 0.9

class ObservedNode():
	''' Represents an observed node (a y value)'''
	def __init__(self, pos, value):
		'''pos is a tuple (row index, col index)
		value is a binary digit
		'''
		self.pos = pos
		self.value = value

	def send_message(self, unobserved_node, theta):
		''' Sends a normalized message to the unobserved_node'''
		message_dicts = unobserved_node.received_messages
		zero_dict = message_dicts[0]
		one_dict = message_dicts[1]

		zero_message = phi(self.value, 0, theta)
		one_message = phi(self.value, 1, theta)

		sum = zero_message + one_message
		zero_message /= sum
		one_message /= sum

		zero_dict[self.pos].append(zero_message)
		one_dict[self.pos].append(one_message)

class UnobservedNode():
	''' Represents an unobserved node (x values)'''
	def __init__(self, pos):
		''' pos is a tuple (row index, col index)
		neighbors contains a list of neighboring UnobservedNodes
		received_message is a list containing two dictionaries.
		The first dictionary holds the messages assuming that x is 0, 
		and the second dictionary holds the messages assuming that x is 1.
		'''
		self.pos = pos
		self.neighbors = []
		self.received_messages = [{}, {}]

	def add_neighbor(self, neighbor):
		''' Adds a neighbor to this Unobserved node'''
		self.neighbors.append(neighbor)
		for message_dict in self.received_messages:
			message_dict[neighbor.pos] = [1]

	def send_messages_to_neighbors(self, theta):
		'''Sends messages to all of its neighbors'''
		for neighbor in self.neighbors:
			if isinstance(neighbor, UnobservedNode):
				self.send_message(neighbor, theta)

	def send_message(self, unobserved_node, theta):
		''' Sends a message to a single neighbor. It sends two 
		messages, one assuming that the neighbor is 0, and the other
		assuming that the neighbor is 1.
		'''
		m_zero = self.compute_message(unobserved_node, 0, theta)
		m_one = self.compute_message(unobserved_node, 1, theta)

		sum = m_zero + m_one
		m_zero /= sum
		m_one /= sum

		unobserved_node.received_messages[0][self.pos].append(m_zero)
		unobserved_node.received_messages[1][self.pos].append(m_one)


	def compute_message(self, unobserved_node, bit, theta):
		''' Computes the message for a given unobserved_node, bit value,
		and theta value.
		'''
		m = 0
		for i in range(2):
			term = phi(i, bit, theta)
			product_of_neighbor_messages = 1
			for n in self.neighbors:
				if n.pos == unobserved_node.pos:
					break
				prev_message = self.received_messages[i][n.pos][-1]
				product_of_neighbor_messages *= prev_message
			term *= product_of_neighbor_messages
			m += term
		return m



	def compute_marginal(self):
		''' After the message values have converged, this returns 0 
		if the last 0 message value is higher, and 1 if the last 1 message
		value is higher. '''
		zero_prod = 1
		one_prod = 1
		for n in self.neighbors:
			if isinstance(n, UnobservedNode):
				zero_prod *= n.received_messages[0][self.pos][-1]
				one_prod *= n.received_messages[1][self.pos][-1]

		if zero_prod > one_prod:
			return 0
		else:
			return 1

class Image():
	''' Represents an image, including the network of observed and unobserved nodes'''
	def __init__(self, file_path):
		'''ob_graph is the network of observed nodes (Y).
		unob_graph is the network of unobserved nodes (X).
		'''
		self.file_path = file_path
		image_file = open(file_path, 'r')
		self.ob_graph = self.init_graph(image_file)
		self.dim = len(self.ob_graph) # Assuming square
		self.unob_graph = self.create_unobserved_graph(self.ob_graph)
		

	def init_graph(self, image_file):
		'''Constructs the network of observed nodes from the image file'''
		graph = []
		for row, line in enumerate(image_file.readlines()):
			new_row = []
			line = line.strip().split()
			for col, bit in enumerate(line):
				bit = int(bit)
				y = ObservedNode((row, col), bit)
				new_row.append(y)
			graph.append(new_row)
		return graph

	def create_unobserved_graph(self, ob_graph):
		'''Constructs the network of unobserved nodes from the observed node graph.'''
		# Fill up the graph, add observed as neighbors
		unob_graph = []
		for row in ob_graph:
			unob_row = []
			for ob in row:
				unob = UnobservedNode(ob.pos)
				unob.add_neighbor(ob)
				unob_row.append(unob)
			unob_graph.append(unob_row)

		# Now, link unobserved nodes with each other
		for unob_row in unob_graph:
			for unob_node in unob_row:
				unob_xpos, unob_ypos = unob_node.pos
				if unob_xpos > 0:
					unob_node.add_neighbor(unob_graph[unob_xpos - 1][unob_ypos])
				if unob_xpos < self.dim - 1:
					unob_node.add_neighbor(unob_graph[unob_xpos + 1][unob_ypos])
				if unob_ypos > 0:
					unob_node.add_neighbor(unob_graph[unob_xpos][unob_ypos - 1])
				if unob_ypos < self.dim - 1:
					unob_node.add_neighbor(unob_graph[unob_xpos][unob_ypos + 1])
		return unob_graph

	def percent_different(self, other_image):
		'''Computes the fraction of pixels from other_image that are different from
		the current image. 
		More precisely, it computes (# pixels that are different) / (total pixels)
		'''
		pixel_count = 0.0
		diff_count = 0.0
		for i in range(len(self.ob_graph)):
			for j in range(len(self.ob_graph)):
				my_pixel = self.ob_graph[i][j].value
				other_pixel = other_image.ob_graph[i][j].value
				if my_pixel != other_pixel:
					diff_count += 1
				pixel_count += 1
		return diff_count / pixel_count




def phi(x1, x2, theta):
	''' The phi function used in the loopy BP algorithm'''
	if x1 == x2:
		return 1 + theta
	else:
		return 1 - theta

def read_input(input_filepath):
	''' Reads the input from a file, and generates a Numpy array'''
	input_file = open(input_filepath, 'r')
	pic = np.genfromtxt(input_file)
	return pic

def loopy_bp(image, theta):
	'''Computes the Loopy BP algorithm, given an instance of the
	Image class, and a theta value. Returns the file path of the 
	denoised image.
	'''
	Y = image.ob_graph
	X = image.unob_graph

	# Every observed node sends a message to its corresponding unobserved node
	t = 0
	while not all_converged(X):
		for row in Y:
			for ob_node in row:
				xpos, ypos = ob_node.pos
				ob_node.send_message(X[xpos][ypos], theta)

		# Now, every unobserved node sends a message to its neighbors
		for row in X:
			for unob_node in row:
				unob_node.send_messages_to_neighbors(theta)

		t += 1
		if t > 20:
			break # Just in case it fails to converge

	denoised_fp = image.file_path + "_denoised"
	denoised_file = open(denoised_fp, 'w')
	for row in X:
		for unob_node in row:
			bit = unob_node.compute_marginal()
			denoised_file.write(str(bit) + " ")
		denoised_file.write('\n')
	denoised_file.close()

	return denoised_fp




def all_converged(X):
	''' Checks if all the message values have converged'''
	all_nodes = [node for row in X for node in row] # Flatten list
	is_converged = True
	for unob_node in all_nodes:
		for bit_dict in unob_node.received_messages:
			for message_list in bit_dict.values():
				if not list_converged(message_list):
					return False
	return True


def list_converged(l):
	''' Checks if a single message value list has converged.'''
	if len(l) < 2:
		return False
	if abs(l[-1] - l[-2]) < .01:
		return True
	else:
		return False

def find_optimal_theta():
	''' Produces the plot finding the optimal Theta, used in the write-up.
	I found that the optimal thetas were the highest, and I decided to use
	0.9 for the bulk of my analysis.
	'''
	THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
	FLIP_PROBS = [0.005, 0.05, 0.25, 0.5]
	noises = []

	disk_fp = image_processing.generate_disk(100, 40)
	original_image = Image(disk_fp)

	noisy_fps = []
	for prob in FLIP_PROBS:
		noisy_fps.append(image_processing.noisify(disk_fp, prob))

	for noisy_image_fp in noisy_fps:
		noise = []
		for theta in THETAS:
			print noisy_image_fp, theta
			noisy_image = Image(noisy_image_fp)
			loopy_bp(noisy_image, theta)
			denoised_image = Image(noisy_image.file_path + "_denoised")
			percent_noise = original_image.percent_different(denoised_image)
			noise.append(percent_noise)
			os.remove(noisy_image.file_path + "_denoised")
		noises.append(noise)

	for noise in noises:
		plt.plot(THETAS, noise)
		plt.xlabel("Theta")
		plt.ylabel("Fraction noise")
	plt.show()

def maximum_noise_plot():
	''' Generates a plot showing the effectiveness of the Loopy BP 
	algorithm for varying mu values, used in the write-up.
	'''
	disk_fp = image_processing.generate_disk(100, 40)
	original_image = Image(disk_fp)

	FLIP_PROBS = [.005, .01, .05, .1, .2, .3, .4, .5]
	noises = []
	for prob in FLIP_PROBS:
		noisy_fp = image_processing.noisify(disk_fp, prob)
		# plt.imshow(np.genfromtxt(noisy_fp), cmap='Greys')
		# plt.show()
		noisy_image = Image(noisy_fp)
		denoised_fp = loopy_bp(noisy_image, THETA)
		denoised_image = Image(denoised_fp)
		fraction_noise = original_image.percent_different(denoised_image)
		print fraction_noise
		noises.append(fraction_noise)

	plt.plot(FLIP_PROBS, noises)
	plt.plot(FLIP_PROBS, FLIP_PROBS)
	plt.xlabel("Mu value (flip probability)")
	plt.ylabel("Fraction noise")
	plt.show()

def print_images():
	''' Prints all the images shown in the write-up.'''
	# Circle with radius 40
	disk_fp = image_processing.generate_disk(100, 40)
	show_image(disk_fp)

	# Circle with mu = 0.01
	noisy_fp = image_processing.noisify(disk_fp, 0.01)
	show_image(noisy_fp)

	# Circle cleaned up, mu = 0.01
	noisy_image = Image(noisy_fp)
	denoised_fp = loopy_bp(noisy_image, THETA)
	show_image(denoised_fp)

	# Circle with mu = 0.1
	noisy_fp = image_processing.noisify(disk_fp, 0.1)
	show_image(noisy_fp)

	# Circle cleaned up, mu = 0.1
	noisy_image = Image(noisy_fp)
	denoised_fp = loopy_bp(noisy_image, THETA)
	show_image(denoised_fp)

	# Square, 60 x 60
	square_fp = image_processing.generate_square(100, 60, 50)
	show_image(square_fp)

	# Square with mu = 0.01
	noisy_fp = image_processing.noisify(square_fp, 0.01)
	show_image(noisy_fp)

	# Square cleaned up, mu = 0.01
	noisy_image = Image(noisy_fp)
	denoised_fp = loopy_bp(noisy_image, THETA)
	show_image(denoised_fp)

	# Equals Sign
	equals_fp = image_processing.generate_equalsign()
	show_image(equals_fp)

	# Equals with mu = 0.01
	noisy_fp = image_processing.noisify(equals_fp, 0.01)
	show_image(noisy_fp)

	# Square cleaned up, mu = 0.01
	noisy_image = Image(noisy_fp)
	denoised_fp = loopy_bp(noisy_image, THETA)
	show_image(denoised_fp)

def show_image(image_fp):
	''' Displays an image, using Numpy and matplotlib.'''
	plt.imshow(np.genfromtxt(image_fp), cmap='Greys')
	plt.show()

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print "Invalid number of arguments -- need to supply an image path and mu value"
		print "python hw6.py <image_filepath> <mu>"
		sys.exit()
	
	file_path = sys.argv[1]
	mu = float(sys.argv[2])

	print "Processing image..."
	original_image = Image(file_path)
	noisy_fp = image_processing.noisify(file_path, mu)
	noisy_image = Image(noisy_fp)

	print "Computing Loopy BP Algorithm..."
	denoised_fp = loopy_bp(noisy_image, THETA)
	denoised_image = Image(denoised_fp)

	show_image(denoised_fp)

	fraction_error = original_image.percent_different(denoised_image)
	print "Finished!"
	print "Error fraction: " + str(fraction_error)


	# IF YOU WANT TO SEE MORE!

	# Generate optimal theta plot
	# find_optimal_theta()

	# Generate maximum noise plot
	# maximum_noise_plot()

	# Generate all example images
	# print_images()















