import numpy as np
from PIL import Image

dataset_path = "C:/Users/sahar/repos/interp/archive/asl_alphabet_train/asl_alphabet_train/A/A1.jpg"
img = Image.open(dataset_path)
kernal_data = np.array(img)
edge_detection_matrix = np.array(([0,-1,0], 
                                 [-1, 8, -1],
                                 [-1, -1, -1]))
final = np.array((0))
x = np.array((3,3))
'''for i in range(200):
    for j in range(200):
        x = np.dot(kernal_data[i,j], edge_detection_matrix)
        for k in range(3):
            final
'''
print(edge_detection_matrix)