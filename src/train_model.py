# current mobilenetv2 with 96 accuracy 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training",
    seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation",
    seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32)

class_names = train_ds.class_names
print("Classes:", class_names)

# ------------------------------

# FIXED: BUILD FUNCTIONAL MODEL
# ------------------------------
inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")

# preprocessing should NOT include separate InputLayers
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

base = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_tensor=x)

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

model.fit(train_ds, validation_data=val_ds, epochs=20)

# this model WILL work with Grad-CAM
model.save("crop_disease_mobilenetv2_FIXED_FINAL.keras")
print("✔ Saved fixed model")
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train', 'Validation'])
plt.show()

















# with MobileNetV2
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # MobileNetV2 MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for MobileNet
# x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# base_model = tf.keras.applications.MobileNetV2(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_mobilenetv2_model.keras")

# print("✔ MobileNetV2 model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("MobileNetV2 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()






#  # with EfficientNetB0
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # EfficientNet MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for EfficientNet
# x = tf.keras.applications.efficientnet.preprocess_input(inputs)

# base_model = tf.keras.applications.EfficientNetB0(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_efficientnet_model.keras")

# print("✔ EfficientNet model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("EfficientNetB0 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()





# # with resnet50
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # RESNET50 MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for ResNet
# x = tf.keras.applications.resnet50.preprocess_input(inputs)

# base_model = tf.keras.applications.ResNet50(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_resnet50_model")

# print("✔ ResNet50 model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("ResNet50 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()




# # with cnn
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # CNN MODEL
# # -----------------------------
# model = models.Sequential([

#     layers.Rescaling(1./255, input_shape=(224,224,3)),

#     layers.Conv2D(32, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(64, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(128, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(256, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.GlobalAveragePooling2D(),

#     layers.Dropout(0.3),

#     layers.Dense(len(class_names), activation="softmax")
# ])

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_cnn_model.keras")
# print("✔ CNN model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("CNN Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Validation"])

# plt.show()





