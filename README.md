# Secondary Mushroom Classification using Machine Learning
 Project Overview: 
This project uses machine learning models to classify mushrooms as edible (0) or poisonous (1) based on their physical features. We used the Secondary Mushroom Dataset from the UCI Machine Learning Repository.

  * Dataset Information: 
Features include: cap shape, cap color, gill attachment, stem color, habitat, etc.
Target variable: class (0 = Edible, 1 = Poisonous)
All features are categorical and properly encoded before model training.

 *Workflow: 
Data Preprocessing
Handled missing values
Label encoding of categorical variables
Standard scaling of features
Model Training
Algorithms used:
Logistic Regression
SVM
Decision Tree
K-Nearest Neighbors (KNN)
Random Forest
Gradient Boosting
AdaBoost
Naive Bayes
MLP Classifier

Model Evaluation
Metrics: Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC Curve
Used train-test split to evaluate performance on unseen data.

Hyperparameter Tuning
Applied GridSearchCV and Pipelines for optimizing selected models.

 Model Performance

Model	Accuracy
Random Forest	99.99%
K-Nearest Neighbors	99.94%
MLP Classifier	99.92%
Decision Tree	99.39%
SVM	97.98%
 Random Forest achieved the best accuracy with almost no misclassifications.

 Random Forest Confusion Matrix
Predicted 0	Predicted 1
Actual 0	5077	5
Actual 1	2	6607\

True Negatives: 5077
True Positives: 6607
False Positives: 5
False Negatives: 2

 Technologies Used
Python 3

Scikit-learn
Pandas
NumPy
Matplotlib
Seaborn

 Conclusion
The Random Forest classifier proved to be the most reliable model, achieving a 99.99% accuracy with excellent precision and recall scores. Machine learning can thus be highly effective for biological classification tasks like mushroom edibility prediction.

 Contact
 Mahalakshmi vs 
 mahalakshmivs1724@gmail.com
Feel free to reach out if you have any questions or suggestions!
