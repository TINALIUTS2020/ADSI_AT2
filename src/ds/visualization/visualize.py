

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
