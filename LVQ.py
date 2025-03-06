import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
class_A = np.random.randn(10, 2) + np.array([2, 2]) 
class_B = np.random.randn(10, 2) + np.array([-2, -2])  

data = np.vstack([class_A, class_B])
labels = np.array([0] * 10 + [1] * 10)  

prototypes = np.array([[2, 2], [-2, -2]])  
prototype_labels = np.array([0, 1])  

alpha = 0.1  
epochs = 20 

for epoch in range(epochs):
    for i in range(len(data)):
        x = data[i]
        true_label = labels[i]

        distances = np.linalg.norm(prototypes - x, axis=1)
        winner_idx = np.argmin(distances)

        if prototype_labels[winner_idx] == true_label:
            prototypes[winner_idx] += alpha * (x - prototypes[winner_idx]) 
        else:
            prototypes[winner_idx] -= alpha * (x - prototypes[winner_idx])  

    alpha *= 0.95

plt.scatter(class_A[:, 0], class_A[:, 1], color='blue', label='Class A')
plt.scatter(class_B[:, 0], class_B[:, 1], color='red', label='Class B')
plt.scatter(prototypes[:, 0], prototypes[:, 1], color='black', marker='X', s=200, label='Prototypes')
plt.legend()
plt.title('LVQ Algorithm Training')
plt.show()
