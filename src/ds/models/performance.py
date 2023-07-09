# RMSE and MAE
def print_reg_perf(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    
    print(f"RMSE {set_name}: {mse(y_actuals, y_preds, squared=False)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")

# For Classification 
def print_log_perf(y_preds, y_actuals, set_name=None):
    """Print the Acc, Precision, Recall, f1 and AUC-ROC for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    print(f"Accuracy {set_name}: {accuracy_score(y_actuals, y_preds)}")
    print(f"Precision {set_name}: {precision_score(y_actuals, y_preds)}")
    print(f"Recall {set_name}: {recall_score(y_actuals, y_preds)}")
    print(f"f1 {set_name}: {f1_score(y_actuals, y_preds)}")
    print(f"AUC-ROC {set_name}: {roc_auc_score(y_actuals, y_preds)}")


# Full set of Matrics (binary)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def evaluate_model_binary(model, X_val, y_val):
    # Make predictions on validation set
    y_val_pred = model.predict(X_val)
    y_val_pred_prob = model.predict_proba(X_val)[:, 1]

    # Calculate evaluation metrics for validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred_prob)

    # Generate ROC curves
    val_fpr, val_tpr, val_thresholds = roc_curve(y_val, y_val_pred_prob)

    # Generate confusion matrices
    val_cm = confusion_matrix(y_val, y_val_pred)

    # Plot ROC curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(val_fpr, val_tpr, label='ROC Curve (AUC = {:.2f})'.format(val_roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Validation Set)')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    # Print evaluation metrics and confusion matrices
    print('Validation Set Metrics:')
    print('Accuracy: {:.4f}'.format(val_accuracy))
    print('Precision: {:.4f}'.format(val_precision))
    print('Recall: {:.4f}'.format(val_recall))
    print('F1-Score: {:.4f}'.format(val_f1))
    print('ROC AUC: {:.4f}'.format(val_roc_auc))
    print('Confusion Matrix (Validation Set):')
    print(val_cm)

# Full set of Matrics (multi)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def evaluate_model_multi(model, X_val, y_val):
    # Make predictions on validation set
    y_val_pred = model.predict(X_val)

    # Calculate evaluation metrics for validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='weighted')
    val_recall = recall_score(y_val, y_val_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    # Generate confusion matrix
    val_cm = confusion_matrix(y_val, y_val_pred)

    # Print evaluation metrics and confusion matrix
    print('Validation Set Metrics:')
    print('Accuracy: {:.4f}'.format(val_accuracy))
    print('Precision: {:.4f}'.format(val_precision))
    print('Recall: {:.4f}'.format(val_recall))
    print('F1-Score: {:.4f}'.format(val_f1))
    print('Confusion Matrix (Validation Set):')
    print(val_cm)


# Chart for model accuracy
def plot_model_accuracy(history,title):
    import matplotlib.pyplot as plt
    plt.title(title)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()

# Chart for model loss
def plot_model_loss(history,title):
    import matplotlib.pyplot as plt
    plt.title(title)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()

# Model- learning curve
def plot_model_learningcurve(history,title):
    import matplotlib.pyplot as plt
    plt.plot(history['accuracy'], label='MSE training')
    plt.plot(history['val_accuracy'], label='MSE validation')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.show()

# Classification report on model predction
def df_classifcation_report (y_encoded, y_prediction, target_names, filename, dest = "../data/external/"):
    """
    y_encoded = y_train_v3_encoded
    y_prediction = v3NN_X_train_v3_predictions
    target_names = target_class
    filename = 'df_clr_v3NN_X_train_v3.csv', 
    dest = ".../data/external"
    """
    from sklearn.metrics import classification_report
    clr = classification_report (y_encoded, 
                                y_prediction, 
                                target_names=target_names, 
                                zero_division=0)
    import pandas as pd
    import re
    # Classification report output

    # Extract class names and metrics using regular expressions
    pattern = r'\s+([\w\s/]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    matches = re.findall(pattern, clr)

    # Create a DataFrame from the extracted information
    clr = pd.DataFrame(matches, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support']).set_index('Class')

    # Convert numeric columns to float
    numeric_cols = ['Precision', 'Recall', 'F1-Score', 'Support']
    clr[numeric_cols] = clr[numeric_cols].astype(float)

    # Print the DataFrame
    print(clr)

    # save classifcaition report as .csv
    clr.to_csv(dest + filename, index=True)
    
    print(clr)
    print(f'csv copy of the classification report is saved at {dest + filename}')

"""
usage: 
df_classifcation_report(
y_encoded = y_train_v3_encoded, 
y_prediction = v3NN_X_train_v3_predictions, 
target_names = target_class,
filename = 'df_clr_v3NN_X_train_v3.csv', 
dest = ".../data/external")
"""

def print_overall_model_metric(y, y_pred, modelname, datasetname):

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Train set
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted',zero_division=0)
    recall = recall_score(y, y_pred, average='weighted',zero_division=0)
    f1_score = f1_score(y, y_pred, average='weighted',zero_division=0)

    # Print the metrics
    print(modelname +" - "+ datasetname +" Accuracy:", accuracy)
    print(modelname +" - "+ datasetname +" Precision:", precision)
    print(modelname +" - "+ datasetname +" Recall:", recall)
    print(modelname +" - "+ datasetname +" F1 Score:", f1_score)