# sentimentanalysis
Sentiment analysis using tokenized comments


![sentimentanalysis](https://github.com/DoollaVenkatasatya/sentimentanalysis/assets/137089784/298e66ba-69e6-457f-90df-7890d1a307b2)

Here's a breakdown of what our code does:

1. **Data Preparation:**
   - We import necessary libraries like `os`, `pandas`, `tensorflow`, and `numpy`.
   - We read the training data from a CSV file called "train.csv" into a pandas DataFrame called `df`.
   - We preprocess the text data using the `TextVectorization` layer from TensorFlow to convert the comments into integer sequences.
   - We split the data into training, validation, and test datasets.

2. **Model Building:**
   - We create a sequential neural network model using Keras.
   - The model includes an embedding layer, a bidirectional LSTM layer, and several dense layers with ReLU activation functions.
   - The output layer has 6 units with a sigmoid activation function, as this is a multi-label classification problem.

3. **Model Training:**
   - We compile the model with 'BinaryCrossentropy' loss and 'Adam' optimizer.
   - We train the model on the training data for 5 epochs, using the validation data for validation.

4. **Model Evaluation:**
   - We plot the training and validation loss over epochs using Matplotlib.
   - We define a function `toxic_check(comment)` that takes a comment as input, preprocesses it, and makes predictions using the trained model.

5. **Model Testing and Metrics:**
   - We use the test dataset to evaluate precision, recall, and accuracy metrics of the model.

6. **Model Deployment:**
   - We save the trained model in an H5 file.
   - We load the saved model and create a Gradio interface to interactively test the model's toxicity check function using user input.

Overall, our code seems to be well-structured, and it follows the standard procedure for building, training, and evaluating a deep learning model for multi-label text classification. Gradio interface provides a simple way for users to input text and get toxicity predictions from our model.
