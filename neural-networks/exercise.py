# Keras uses the Sequential model for linear stacking of layers.
# That is, creating a neural network is as easy as (later)
# defining the layers!
from tensorflow.keras.models import Sequential
# Everything we've talked about in class so far is referred to in
# Keras as a "dense" connection between layers, where every input
# unit connects to a unit in the next layer
# We will go over specific activation functions throughout the class.
from tensorflow.keras.layers import Dense
# SGD is the learning algorithm we will use
from tensorflow.keras.optimizers import SGD


def build_one_output_model():
    model = Sequential()

    ### YOUR CODE HERE ###
    # Add a input hidden layer with appropriate input dimension
    # 1+ lines
    model.add(Dense(1, input_dim = 2, activation='relu'))
   

    # Add a final output layer with 1 unit 
    # 1 line
    model.add(Dense(1, activation='sigmoid'))
    


    ######################

    sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)  #Stochastic gradient descent
    model.compile(loss="binary_crossentropy", optimizer=sgd)
    return model


def build_classification_model():
    model = Sequential()

    ### YOUR CODE HERE ###
    # First add a fully-connected (Dense) hidden layer with appropriate input dimension
    model.add(Dense(10, input_dim = 2, activation='relu'))
    
    # Now our second hidden layer 
    model.add(Dense(5, activation='relu'))
    


    # Finally, add a readout layer
    model.add(Dense(2, activation='softmax'))

    

    ######################

    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=["accuracy"])
    return model


def build_final_model():
    model = Sequential()
    ### YOUR CODE HERE ###
    from tensorflow.keras.layers import Dropout
    model.add(Dense(50, input_dim=50, activation='relu')) #hidden layer
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu')) #hidden layer
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu')) #hidden layer
    model.add(Dropout(0.2))
    model.add(Dense(80, activation='relu')) #hidden layer
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu')) #hidden layer
    model.add(Dense(2, activation='softmax')) #output later
    
   
    ######################
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    # we'll have the categorical crossentropy as the loss function
    # we also want the model to automatically calculate accuracy

    return model

def logistic_regression_model():
    
    ### YOUR CODE HERE ###
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C = 0.1, penalty = 'l2', solver = 'liblinear')
    
    return model


def random_forest_model():
    
    ### YOUR CODE HERE ###
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=26)
    
    return model