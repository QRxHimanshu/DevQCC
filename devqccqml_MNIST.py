


from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
# features = iris_data.data
# labels = iris_data.target
# features, test_features, labels, test_labels = train_test_split(
#     features, labels, train_size=0.05, random_state=algorithm_globals.random_seed
# )


import tensorflow as tf
import numpy as np
# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_filter = np.where((train_labels == 0) | (train_labels == 1))
test_filter = np.where((test_labels == 0) | (test_labels == 1))

# Normalize pixel values to range between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)
features = train_images_flat[train_filter]
labels = train_labels[train_filter]
features, test_features, labels, test_labels = train_test_split(
    features, labels, train_size=0.05, random_state=algorithm_globals.random_seed
)

from sklearn.decomposition import PCA

features = PCA(n_components=6).fit_transform(features)
from sklearn.preprocessing import MinMaxScaler

features = MinMaxScaler().fit_transform(features)
print(features.shape)
print(labels.shape)

"""Let's see how our data looks. We plot the features pair-wise to see if there's an observable correlation between them."""

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = 123
# features, test_features, labels, test_labels = train_test_split(
#     features, labels, train_size=0.05, random_state=algorithm_globals.random_seed
# )
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.7, random_state=algorithm_globals.random_seed
)

"""We train a classical Support Vector Classifier from scikit-learn. For the sake of simplicity, we don't tweak any parameters and rely on the default values."""

from sklearn.svm import SVC

svc = SVC()
_ = svc.fit(train_features, train_labels)  # suppress printing the return value

"""Now we check out how well our classical model performs. We will analyze the scores in the conclusion section."""

train_score_c4 = svc.score(train_features, train_labels)
test_score_c4 = svc.score(test_features, test_labels)

print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")


from qiskit.circuit.library import ZFeatureMap

num_features = features.shape[1]

feature_map = ZFeatureMap(feature_dimension=num_features, reps=1)
# feature_map.decompose().draw(output="mpl", style="clifford", fold=20)

"""If you look closely at the feature map diagram, you will notice parameters `x[0], ..., x[3]`. These are placeholders for our features.

Now we create and plot our ansatz. Pay attention to the repetitive structure of the ansatz circuit. We define the number of these repetitions using the `reps` parameter.
"""

from qiskit.circuit.library import RealAmplitudes

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
# ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
# print(ansatz.decompose().data)

"""This circuit has 16 parameters named `θ[0], ..., θ[15]`. These are the trainable weights of the classifier.

We then choose an optimization algorithm to use in the training process. This step is similar to what you may find in classical deep learning frameworks. To make the training process faster, we choose a gradient-free optimizer. You may explore other optimizers available in Qiskit.
"""

from qiskit_algorithms.optimizers import COBYLA
mxiter = 100
optimizer = COBYLA(maxiter=mxiter)

"""In the next step, we define where to train our classifier. We can train on a simulator or a real quantum computer. Here, we will use a simulator. We create an instance of the `Sampler` primitive. This is the reference implementation that is statevector based. Using qiskit runtime services you can create a sampler that is backed by a quantum computer."""

from qiskit.primitives import Sampler

sampler = Sampler()

"""We will add a callback function called `callback_graph`. `VQC` will call this function for each evaluation of the objective function with two parameters: the current weights and the value of the objective function at those weights. Our callback will append the value of the objective function to an array so we can plot the iteration versus the objective function value. The callback will update the plot at each iteration. Note that you can do whatever you want inside a callback function, so long as it has the two-parameter signature we mentioned above."""

from matplotlib import pyplot as plt1
from matplotlib import pyplot as plt
from IPython.display import clear_output

objective_func_vals = []
plt1.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.figsize"] = (12, 6)
weight_vals= []
train_score_list = []

def callback_graph(weights, obj_func_eval):
    print("Objective value after iteration :",obj_func_eval)
    clear_output(wait=True)
    # train_score = vqc.score(train_features, train_labels)
    # train_score_list.append(train_score)
    objective_func_vals.append(obj_func_eval)
    weight_vals.append(weights)


import time
from qiskit_machine_learning.algorithms.classifiers import VQC

# plist = [0.86111111,0.33333333,0.86440678,0.75]
# feature_map = feature_map.decompose()
# params = feature_map.parameters
# feature_map.assign_parameters(dict(zip(params, plist)))

# feature_map.draw(output="mpl", style="clifford", fold=20)

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map.decompose(),
    ansatz=ansatz.decompose(),
    optimizer=optimizer,
    callback=callback_graph,
)

# clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start



print(f"Training time: {round(elapsed)} seconds")

"""Let's see how the quantum model performs on the real-life dataset."""

train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

vqc.save("QMLResults/mnist/binary0_1new/mnist_binarynew_multiclass.model")

# # Plot the training accuracy curve
# plt.figure()
# plt.plot(range(1, mxiter+ 1), train_score_list, label='Training Accuracy')
# plt.xlabel('Iteration')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy Curve')
# plt.legend()
# plt.savefig("Accuracy.png")

plt1.title("Distributed QML training using DevQCC-QML ")
plt1.xlabel("Iteration")
plt1.ylabel("Loss")
plt1.plot(range(len(objective_func_vals)), objective_func_vals)
plt1.savefig("QMLResults/mnist/binary0_1new/binary_obj_iter.png")

# print(weight_vals)
from sklearn.metrics import *
test_predictions = vqc.predict(test_features)

import numpy as np

# Confusion Matrix
cm = confusion_matrix(test_labels, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for VQC")
plt.savefig("QMLResults/mnist/binary0_1new/binary_confusion_matrix.png")



if len(np.unique(labels)) == 2:
    # Get raw predictions as scores
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(test_labels, test_predictions)

    # Plot precision-recall curve
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for VQC')
    plt.legend(loc='lower left')
    plt.savefig("QMLResults/mnist/binary0_1new/binary_precision_recall_curve.png")
else:
    print("Precision-recall curve plotting is currently supported only for binary classification.")

if len(np.unique(labels)) == 2:
    # Get raw predictions as scores
    test_scores = test_predictions  # Use predictions directly for ROC curve
    
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("QMLResults/mnist/binary0_1new/binary_roc_curve.png")
else:
    print("ROC curve plotting is currently supported only for binary classification.")

