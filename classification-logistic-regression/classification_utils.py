def accuracy(truth_values, predicted_values):
    """
    Computes accuracy
    :param truth_values: true values
    :param predicted_values: predictions

    :return: accuracy
    """
    TP = 0
    TN = 0
    size = truth_values.shape[0]
    
    for i in range(size):
        if (truth_values[i] == predicted_values[i] and truth_values[i] == 1):
            TP += 1
        elif (truth_values[i] == predicted_values[i] and truth_values[i] == 0):
            TN += 1
 
    accuracy = (TP+TN)/size
 
    ######################

    return accuracy

def precision(truth_values, predicted_values):
    """
    Computes precision
    :param truth_values: true values
    :param predicted_values: predictions

    :return: precision
    """
    TP = 0
    FP = 0
    size = truth_values.shape[0]
        
    for i in range(size):
        if (truth_values[i] == predicted_values[i] and truth_values[i] == 1):
            TP += 1
        elif (truth_values[i] != predicted_values[i] and truth_values[i] == 0):
            FP += 1
 
    precision = TP/(TP+FP)
 
    ######################

    return precision


def recall(truth_values, predicted_values):
    """
    Computes recall
    :param truth_values: true values
    :param predicted_values: predictions

    :return: recall
    """
    TP = 0
    FN = 0
    size = truth_values.shape[0]
    
    for i in range(size):
        if (truth_values[i] == predicted_values[i] and truth_values[i] == 1):
            TP += 1
        elif (truth_values[i] != predicted_values[i] and truth_values[i] == 1):
            FN += 1
 
    recall = TP/(TP+FN)
 
    ######################

    return recall

