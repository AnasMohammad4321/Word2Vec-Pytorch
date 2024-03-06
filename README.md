# Word2Vec Algorithm Implementation

## Introduction
This project implements the Word2Vec algorithm using the Amazon dataset. Word2Vec is a popular technique in natural language processing for learning word embeddings. These embeddings capture semantic relationships between words, making them useful for various NLP tasks such as sentiment analysis, machine translation, and named entity recognition.

## Implementation Details
The implementation is done in Python using the PyTorch framework. Key components and libraries used in the project include:

- PyTorch: Deep learning framework for building and training neural networks.
- NumPy: Library for numerical operations in Python.
- Gensim: Library for topic modeling and document similarity, used for saving the Word2Vec model.
- NLTK: Library for natural language processing tasks, used for tokenization.
- tqdm: Library for displaying progress bars during training.
- Weights & Biases (wandb): Tool for experiment tracking.
- SciPy: Library for scientific computing, used for cosine similarity calculation.
- argparse: Library for command-line argument parsing.

## Components
### 1. Corpus Class
- Tokenizes the input text and preprocesses it.
- Generates negative sampling table for efficient training.
- Provides methods for loading data and generating negative samples.

### 2. RandomNumberGenerator Class
- Generates random numbers for data augmentation and negative sampling.

### 3. Word2VecDataset Class
- Custom PyTorch dataset for training Word2Vec model.

### 4. Word2Vec Class
- Defines the architecture of the Word2Vec model.

### 5. Training
- Defines hyperparameters such as batch size, embedding size, learning rate, etc.
- Trains the Word2Vec model using the provided dataset.

### 6. Evaluation
- Computes cosine similarity between words to find similar words in the trained embedding space.

### 7. Save Function
- Saves the trained Word2Vec model and corpus object for later use.

## Usage
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Adjust hyperparameters and file paths according to your requirements.
3. Run the script to train the Word2Vec model.
4. Evaluate the trained model using the provided evaluation functions.
5. Save the model and corpus for future use.

## Results
- After training, the Word2Vec model can be used to find similar words, analyze word relationships, or as input for downstream NLP tasks.

## Conclusion
This project demonstrates the implementation of the Word2Vec algorithm using the Amazon dataset. By training word embeddings, we can capture semantic relationships between words, which are useful for various natural language processing tasks. The trained model can be saved and reused for future applications.

## References
- Original Word2Vec Paper: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- Gensim Documentation: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
- NLTK Documentation: [https://www.nltk.org/](https://www.nltk.org/)

Feel free to explore and modify the code according to your needs! If you have any questions or suggestions, please don't hesitate to reach out.

## License
This project is licensed under the [MIT License](LICENSE).
