import pandas as pd

def calc_class_averages(data, labels, label_col):
    """
    Calculate class averages for the unique labels in the dataset.
    
    Raises:
        ValueError: If the number of rows in data and labels do not match.
    """
    
    if data.shape[0] != labels.shape[0]:
        raise ValueError("The number of rows in data and labels must match.")
    
    select_labels = labels[label_col]
    combined = pd.concat([select_labels, data], axis=1)
    reduced = pd.pivot_table(combined,
                             index = label_col,
                             values = combined.columns[1:],
                             aggfunc = 'mean',
                             )
    
    return reduced
    
