

def plot_column_dist(data, column_name, top_n=None):
    """Plot column dist

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    column_name : str
        Name of the column to plot
    top_n: int
        if leave blank, then display all values

    Returns
    -------
    chart
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    column_data = data[column_name]

    # Calculate the counts of each value in the specified column
    value_counts = column_data.value_counts()

    # Create a table of labels and counts
    table = pd.DataFrame({column_name: value_counts.index, 'count': value_counts.values})

    # Sort the table by count in descending order
    sorted_table = table.sort_values('count', ascending=False)

    if top_n is not None:
        # Select the top n values
        sorted_table = sorted_table[:top_n]

    plt.figure(figsize=(20, 6))

    # Define the desired width between bars
    bar_width = 0.6

    # Plot the bars using the table data
    plt.bar(range(len(sorted_table)), sorted_table['count'], align='center', width=0.8)

    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title(f'{column_name} Distribution')

    # Add data labels to the bars
    for i, count in enumerate(sorted_table['count']):
        plt.text(i, count, f"{count:,.0f}", ha='center', va='bottom', fontsize=8, rotation=90)

    plt.xticks(range(len(sorted_table)), sorted_table[column_name], rotation=90)

    # Adjust the x-axis limits to shorten the distance between the left edge and the first x-label
    plt.xlim(-0.8, len(sorted_table) - 0.8)

    # Adjust the spacing at the bottom of the plot
    plt.subplots_adjust(bottom=0.3)
    plt.show()




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