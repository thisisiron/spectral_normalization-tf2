# spectral_normalization-tf2
Spectral Normalization implemented as Tensorflow 2.0

## Run Test Code
- [ ] Convert simpole conv2d model to DCGAN

This is currently a test code using a simple image classification model.
```
python main.py
```

## Algorithm
![](./images/algorithm.png)

## How to use
1. Sequential API 
```
model = models.Sequential()
model.add(SpectralNormalization(layers.Conv2D(32, (3, 3), activation='relu')))
...
```

2. Functional API
```
inputs = layers.Input(shape=(28,28,1))
x = SpectralNormalization(layers.Conv2D(32, (3, 3), activation='relu'))(inputs)
...
````

3. Custom Layer Method 
```
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        self.conv2DSN = SpectralNormalization(layers.Conv2D(32, (3, 3), activation='relu'))
        ...
    
    def call(self, inputs):
        x = self.conv2DSN(inputs)
        ...
```

## Rerference
[Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
