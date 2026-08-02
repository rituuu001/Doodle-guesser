import numpy as np
import matplotlib.pyplot as plt

data = np.load("C:/projects/Doodle-guesser/data/raw/cup.npy")
sample = data[0].reshape(28, 28)
print(sample.min(), sample.max())
plt.imshow(sample, cmap="gray")
plt.show()