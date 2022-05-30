import math

import networkx as nx
import numpy as np
import scipy.optimize as spo
from matplotlib import pyplot as plt
from scipy.sparse import csgraph
from sklearn.decomposition import PCA


def function_to_minimize(Y, L, d):
	n = int(math.sqrt(len(L)))
	L = np.resize(L, (n, n))
	Y = np.resize(Y, (n, d))
	return np.trace(np.transpose(Y) @ L @ Y)


def restriction(Y, D, d):
	Y = np.resize(Y, (np.size(D, 0), d))
	D = np.diag(D)
	left = (np.transpose(Y) @ D) @ Y
	req_matrix = left - np.identity(d)
	res = np.linalg.norm(req_matrix)

	return -res


def optimize_y(graph, d):
	n = len(graph)
	y_start = np.random.rand(n, d)
	diagonal = np.sum(graph, 0)
	L = csgraph.laplacian(graph)
	restr = {'type': 'eq', 'fun': restriction, 'args': (diagonal, d)}
	res = spo.minimize(
		function_to_minimize,
		y_start,
		args=(np.reshape(L, -1), d),
		constraints=restr)
	return res.x


def check_barbell():
	n = 11
	path_len = 0
	graph = nx.barbell_graph(n, path_len)
	plt.figure(figsize=(10, 6))
	nx.draw(graph, node_color='lightblue',
	        with_labels=True,
	        node_size=500)
	plt.savefig("barbell.png")
	n = nx.number_of_nodes(graph)
	alignment_matrix = nx.to_numpy_array(graph)
	for i in range(2, n + 1, 10):
		y = optimize_y(alignment_matrix, i)
		y = np.reshape(y, (n, i))
		pca = PCA(n_components=2)
		y = pca.fit_transform(y)
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_xlabel('Principal Component 1', fontsize=15)
		ax.set_ylabel('Principal Component 2', fontsize=15)
		ax.set_title(f'2 component PCA for barbell graph\ngiven {i} starting components', fontsize=20)
		ax.scatter(y[:, 0], y[:, 1], s=50)
		ax.grid()
		fig.savefig(f"Barbell from {i} components.png")


def check_cycle(n):
	graph = nx.cycle_graph(n)
	plt.figure(figsize=(10, 6))
	nx.draw(graph, node_color='lightblue',
	        with_labels=True,
	        node_size=500)
	plt.savefig("cycle.png")
	alignment_matrix = nx.to_numpy_array(graph)
	for i in range(2, n + 1, 5):
		y = optimize_y(alignment_matrix, i)
		y = np.reshape(y, (n, i))
		pca = PCA(n_components=2)
		y = pca.fit_transform(y)
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_xlabel('Principal Component 1', fontsize=15)
		ax.set_ylabel('Principal Component 2', fontsize=15)
		ax.set_title(f'2 component PCA for cycle of length {n}\ngiven {i} starting components', fontsize=20)
		ax.scatter(y[:, 0], y[:, 1], s=50)
		ax.grid()
		fig.savefig(f"Cycle of length {n} from {i} components.png")


def check_random():
	n = 16
	graph = nx.gnp_random_graph(n, 3 / n)
	plt.figure(figsize=(10, 6))
	nx.draw(graph, node_color='lightblue',
	        with_labels=True,
	        node_size=500)
	plt.savefig("random.png")
	alignment_matrix = nx.to_numpy_array(graph)
	print(alignment_matrix)
	for i in range(2, n + 1, 7):
		y = optimize_y(alignment_matrix, i)
		y = np.reshape(y, (n, i))
		pca = PCA(n_components=2)
		y = pca.fit_transform(y)
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_xlabel('Principal Component 1', fontsize=15)
		ax.set_ylabel('Principal Component 2', fontsize=15)
		ax.set_title(f'2 component PCA for random graph\n with {n} nodes given {i} starting components', fontsize=20)
		# print(f"{y[:,0] = }, {y[:,1] = }")
		ax.scatter(y[:, 0], y[:, 1], s=50)
		ax.grid()
		fig.savefig(f"Random with {n} nodes from {i} components.png")


def check_cycles():
	check_cycle(12)
	check_cycle(17)


def main():
	# check_barbell()
	# check_cycles()
	check_random()


if __name__ == '__main__':
	main()
