# 🌸 Flower Detection with Deep Learning (SE4050) 🌸

## IT21302480 - Dilshan.O.A.P.
## IT21195570 - Herath H.M.K.C.B 
## IT21298912 - Yapa Y.M.T.N.S
## IT19985428 - Rathnasiri K.D.M.M


This project implements **Flower Detection** using deep learning models: **ResNet152**, **VGG16**, **EfficientNet**, and **Xception**.
---

## 🏗️ DL Project Overview

We trained models on a flower dataset consisting of 5 flower classes:
- 🌼 Daisy
- 🌻 Dandelion
- 🌹 Rose
- 🌸 Sunflower
- 🌷 Tulip

Models used:
1. **ResNet152** (152 layers)
2. **VGG16** (16 layers)
3. **EfficientNet**
4. **Xception**

---

## 🏋️‍♂️ Model Training

To train any model, ensure the dataset is in place, then use this general command structure:

```python
# Example for training ResNet model
model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Common layers for all models
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

# Model creation
final_model = Model(inputs=model.input, outputs=predictions)

# Compile and train
final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model.fit(train_data, epochs=10, validation_data=val_data)
```

## 🧪 Evaluation

Evaluate the model:

``` python
accuracy = final_model.evaluate(test_data)[1] * 100
print(f"Test Accuracy: {accuracy}%")
```

## 🔮 Prediction

Make predictions on a new image:

``` python
img = image.load_img('/path/to/image.jpg', target_size=(224, 224))
img_array = np.expand_dims(image.img_to_array(img), axis=0)
prediction = final_model.predict(img_array)
print(f"Predicted Class: {np.argmax(prediction)}")
```

## 💾 Save & Load
### Save the model:

``` python
final_model.save('flower_model.h5')
```

### Load the model:

``` python
model = load_model('flower_model.h5')
```

## 🎯 Results

Expected accuracy for each model:

| Model        | Accuracy (%) |
|--------------|--------------|
| ResNet152    | ~96.5        |
| VGG16        | ~94.2        |
| EfficientNet | ~97.1        |
| Xception     | ~96.0        |

These results were obtained by training each model on the flower dataset for 10 epochs.


### 🌸 Happy Flower Detection!

This is the **best compact** version, focusing only on essential content for training, evaluating, and predicting flower detection.
