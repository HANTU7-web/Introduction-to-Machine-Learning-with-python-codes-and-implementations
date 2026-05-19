import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.datasets import fetch_lfw_people
import pandas as pd
import seaborn as sns

people = fetch_lfw_people(min_faces_per_person = 20, resize = 0.7)
image_shape = people.images[0].shape

fig , axes = plt.subplots(2, 5, figsize=(15,8),
                          subplot_kw={'xticks':[], 'yticks':[]})

for target , images, ax in zip(people.target , people.images , axes.ravel()):
    ax.imshow(images)
    ax.set_title(people.target_names[target])
plt.show()
