# Machine-Learning-Assignment 1 :-
Titled : "Recognize a Digit using Machine Learning"

code block 1 :
import tensorflow as tf: Loads TensorFlow library, which we use for building and training machine learning models.
import matplotlib.pyplot as plt: Used for visualizing images and results using plots.

code block 2 :
tf : is the alias for TensorFlow (you must have imported it earlier with import tensorflow as tf).
tf.keras : is TensorFlow’s high-level API for building and training models.
tf.keras.datasets : is a module that contains a few small datasets for quick experiments.
mnist : is one of those datasets: it contains 28x28 pixel grayscale images of handwritten digits (0 through 9).

code block 3 :
mnist.load_data() :
            This function loads the dataset and returns two sets of data:
                      Training set (used to train the model)
                      Test set (used to evaluate how well the model performs)
            x_train and y_train for training.
            x_test and y_test for testing.

code block 4 :            
x_train = x_train / 255.0, x_test = x_test / 255.0 : These lines normalize the pixel values of the images from 0–255 to 0–1, which helps the neural network learn more efficiently.

code block 5 :
model = tf.keras.models.Sequential([ : Starts building a sequential model — a linear stack of layers, where data flows from one layer to the next in order.
tf.keras.layers.Flatten(input_shape=(28, 28)), : Flattens the 28x28 pixel image into a 1D array of 784 values so it can be processed by the dense (fully connected) layers.
tf.keras.layers.Dense(128, activation='relu'), : Adds a fully connected hidden layer with 128 neurons using ReLU activation, which introduces non-linearity to help the model learn complex patterns.
tf.keras.layers.Dense(10, activation='softmax') : Adds the output layer with 10 neurons (one for each digit 0–9) using softmax activation, which outputs probabilities for classification.

code block 6 :
model.compile(...) : Prepares (compiles) the model for training by specifying how it should learn.
optimizer='adam' : Uses the Adam optimizer, which adjusts weights efficiently during training to minimize error. It's widely used because it combines the best parts of SGD and RMSProp.
loss='sparse_categorical_crossentropy' : Specifies the loss function used to measure the difference between predicted and actual labels.
            "Sparse" means that the labels are integers (e.g., 3, 5) instead of one-hot encoded vectors.
            This is perfect for multi-class classification like digit recognition (0–9).
metrics=['accuracy'] : Tells the model to monitor accuracy during training and evaluation, so you can see how well it’s doing.

code block 7 :
model.fit(...) : This function trains the model using the training data.
x_train : The input images (28x28 grayscale digits) used for training.
y_train : The correct labels (digits 0–9) corresponding to each training image.
epochs=5 : The model will go through the entire training dataset 5 times. Each complete pass through the dataset is called an epoch.

code block 8 :
model.evaluate(x_test, y_test) : This runs the trained model on the test data (which it hasn’t seen before) to check its performance.
            It returns two values:
                    test_loss: How well (or poorly) the model performed.
                    test_acc: The accuracy of the model on the test dataset.
test_loss, test_acc = ...  :  This line stores the two values returned by evaluate() into variables test_loss and test_acc.
print('\nTest Accuracy:', test_acc) : This prints the accuracy of the model on unseen data to the output.

code block 9 :
model.predict(x_test) : This uses the trained model to make predictions on the test images (x_test).
predictions : This stores the output probabilities for each test image. Each prediction is a list of 10 numbers (for digits 0–9), showing how confident the model is for each digit.

code block 10 :
plt.imshow(x_test[0], cmap=plt.cm.binary) :
            This displays the first test image (x_test[0]) as a grayscale image (binary colormap).
            plt.imshow() is from matplotlib.pyplot, used to display images.
plt.title(f"Model Prediction: {tf.argmax(predictions[0])}, Actual: {y_test[0]}")
            This sets the title above the image.
            tf.argmax(predictions[0]) finds the digit with the highest probability from the model’s prediction.
            y_test[0] is the actual label for that image.
            The title shows both what the model predicted and what the correct digit is.
plt.axis('off') :
            This hides the X and Y axes for a cleaner look.   
plt.show() :
            This actually displays the image with the title and no axes.            
