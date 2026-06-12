# Deep Learning from Scratch

Implémentation de réseaux de neurones from scratch en Python/NumPy, sans framework de deep learning.

## Notebooks

| Notebook | Description |
|---|---|
| `first_neuron.ipynb` | Neurone artificiel — sigmoïde, log-loss, descente de gradient, frontière de décision animée, classification chats/chiens |
| `neural_network.ipynb` | Réseaux multicouches — `Neural_Network` (1 couche cachée) et `DeepNeuralNetwork` (architecture configurable) |

## Concepts couverts

- Propagation avant et rétropropagation
- Log-loss (binary cross-entropy)
- Descente de gradient
- Activation sigmoïde
- Visualisation animée de la frontière de décision
- Classification d'images (64×64, chats vs chiens)

## Dataset

Le dossier `datasets/` contient les sets d'entraînement et de test d'images 64×64 (chats vs chiens).

## Requirements

```
numpy
matplotlib
scikit-learn
tqdm
h5py
```
