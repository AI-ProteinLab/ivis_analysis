import numpy as np

def get_probability(X, bin_length):
	shannon = []

	for index in range(X.shape[1]):
		data = X[:, index]
		if max(data) == 0:
			continue
		step = (max(data) - min(data)) * (1 + 1e-5) / bin_length
		count = [0 for _ in range(bin_length)]

		for num in data:
			count[int((num - min(data)) // step)] += 1
		probability = [num / X.shape[0] for num in count]

		entropy = sum([-np.log2(num) for num in probability if num != 0])
		
		shannon.append(entropy)

	return sum(shannon), sum(shannon) / len(shannon)

