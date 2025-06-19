# 🧠 Kaggle Digit Recognizer: CNN vs Traditional ML Models

This project solves the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) challenge using two distinct approaches: a custom Convolutional Neural Network (CNN) and traditional Machine Learning models. The goal is to classify handwritten digits from the MNIST dataset and compare the performance of deep learning and classic ML algorithms.

## 🎯 Project Objective

- Build a Convolutional Neural Network (CNN) from scratch using TensorFlow.
- Train two traditional ML models: Support Vector Machine (SVM) and K-Nearest Neighbors (KNN).
- Evaluate and compare model performance.
- Submit predictions to Kaggle and analyze leaderboard results.

## 🛠️ Tasks Breakdown

### ✅ Task 1: Custom CNN Model

- **Framework**: TensorFlow
- **Preprocessing**:
  - Normalize pixel values to [0, 1]
  - Reshape images to (28, 28, 1)
- **Architecture**:
  - Conv2D → MaxPooling → Dropout → Dense
- **Training**:
  - Optimizer: Adam
  - Epochs: Tuned experimentally
- **Evaluation**:
  - Accuracy and loss on training, validation, and test sets
  - Predictions saved as `submission_cnn.csv`

### ✅ Task 2: Traditional ML Models

- **Models Used**:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **Preprocessing**:
  - Flatten 28x28 images to 784-dimensional vectors
  - Normalize pixel values
- **Training**:
  - SVM: Tuned C and kernel
  - KNN: Tuned n_neighbors
- **Evaluation**:
  - Accuracy and F1-score
  - Predictions saved as `submission_ml.csv`

---

## 📊 Report Highlights

- **Model Architectures**
  - CNN: Multiple Conv2D, MaxPooling, Dropout layers followed by Dense output
  - ML: SVM with RBF kernel and KNN with tuned neighbors
- **Hyperparameters**
  - CNN: Learning rate, batch size, dropout rate, epochs
  - ML: n_neighbors (KNN), C and kernel (SVM)
- **Performance Metrics**
  - Training/validation/test accuracy
  - Confusion matrices and classification reports
  - Visuals: Training curves, confusion matrix heatmaps
- **Error Analysis**
  - Misclassified digit samples
  - Model-specific error trends
- **Comparative Summary**
  - CNN achieved higher accuracy with more training time and GPU usage
  - ML models trained faster but performed slightly lower on accuracy
  - Deployment Recommendation: CNN preferred for accuracy, ML for quick and low-resource deployment

---

## 🏆 Kaggle Submission

- ✅ CNN Accuracy: (as per leaderboard)
- ✅ ML Model Accuracy: (as per leaderboard)
- 📸 Screenshots of leaderboard included

---

## 📚 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Matplotlib / Seaborn / Plotly
- Pandas / NumPy

---

## 📌 How to Run

1. Clone the repository
2. Run notebooks:
   - `cnn_model.ipynb` for CNN model
   - `ml_models.ipynb` for SVM and KNN models
3. Submit `submission_cnn.csv` or `submission_ml.csv` to Kaggle

---

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer)
- MNIST dataset
