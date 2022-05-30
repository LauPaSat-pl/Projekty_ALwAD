import random

import matplotlib.pyplot as plt
import numpy as np


def encode(v, g, field):
	return (np.transpose(g)).dot(v) % field


def hamming_distance(str1, str2):
	assert len(str1) == len(str2)
	count = 0
	for i in range(len(str1)):
		if str1[i] != str2[i]:
			count += 1
	return count


def minimize_hamming_distance(c, b, v):
	min_dist = 999
	vectors_with_given_len = []
	for vector in c:
		d = hamming_distance(v, vector)
		if d < min_dist:
			min_dist = d
			vectors_with_given_len = [vector]
		elif d == min_dist:
			vectors_with_given_len.append(vector)
	w = random.choice(vectors_with_given_len)
	return w[:len(b)]


def transmission_sim(encoded_vector, field):
	noise_vector = np.zeros(len(encoded_vector), dtype=np.int64)
	for i in range(len(noise_vector)):
		temp = random.random()
		if temp > 0.95:
			noise_vector[i] = 3
	return (encoded_vector + noise_vector) % field


def vector_generation(base, field):
	all_vectors = []
	for i in range(field):
		for j in range(field):
			for k in range(field):
				for m in range(field):
					all_vectors.append((i * base[0] + j * base[1] + k * base[2] + m * base[3]) % field)
	return all_vectors


def exercise_6(base: list, field: int):
	all_vectors = []
	for i in range(field):
		for j in range(field):
			for k in range(field):
				all_vectors.append((i * base[0] + j * base[1] + k * base[2]) % field)
	return all_vectors


def exercise_7(b, v, field):
	g = np.array(b)
	c = exercise_6(b, field)
	r = minimize_hamming_distance(c, b, v)
	print(g)
	return r


def exercise_8(generating_matrix, field):
	a = np.random.default_rng().integers(5, size=(4, 10))
	a_norm = a / 4
	base = list(generating_matrix)
	code = vector_generation(base, field)
	result = np.zeros((4, 10), dtype=np.int64)
	plt.matshow(a_norm)
	plt.colorbar()
	plt.savefig("a_norm.png")

	for i in range(len(a[0])):
		encoded_vector = encode(a[:, i], generating_matrix, field)
		sent_vector = transmission_sim(encoded_vector, field)
		decoded_vector = minimize_hamming_distance(code, base, sent_vector)
		result[:, i] = decoded_vector

	plt.matshow(result / 4)
	plt.colorbar()
	plt.savefig("res.png")
	print(a-result)
	return result


def main():
	b = [np.array([1, 0, 0, 2, 4]), np.array([0, 1, 0, 1, 0]), np.array([0, 0, 1, 5, 6])]
	v = np.array([1, 2, 3, 4, 5])
	g = np.array(
		[[1, 0, 0, 0, 0, 4, 4, 2, 0, 1, 1], [0, 1, 0, 0, 0, 3, 0, 2, 2, 1, 0], [0, 0, 1, 0, 0, 2, 0, 1, 1, 1, 1],
		 [0, 0, 0, 1, 1, 0, 0, 0, 4, 3, 0]])

	np.savetxt('zad.6', np.array(exercise_6(b, 7)), fmt='%.0f', newline=']\n[', delimiter=', ')
	print(exercise_7(b, v, 7))
	exercise_8(g, 5)


if __name__ == '__main__':
	main()
