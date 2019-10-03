# spectral_normalization-tf2
Spectral Normalization implemented as Tensorflow 2.0

## Run Test Code
TODO: main.py -> simple DCGAN model
```
python main.py
```

## Algorithm
![](./images/algorithm.png)

## How to use
1. Sequential Method
```
model = models.Sequential()
model.add(SpectralNormalization(layers.Conv2D(32, (3, 3), activation='relu')))
...
```
2. Class Method
```
class Custom(tf.keras.layers.Layer):
    def __init__(self):
        self.conv2DSN = SpectralNormalization(layers.Conv2D(32, (3, 3), activation='relu'))
        ...
```

## Rerference
[Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
