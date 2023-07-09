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
