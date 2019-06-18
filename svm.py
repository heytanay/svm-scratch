try:
    import numpy as np
    from matplotlib import pyplot as plt
    # %matplotlib inline
    print("Numpy and Matplotlib imported")
except:
    print("An error has occurred, couldnot import numpy and matplotlib")


# This is just some random data that form some nice plot points 
X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

# Just some random Y data points
y = np.array([-1,-1,1,1,1])

# For an SVM, we are going to use something called a Hinge Loss and then we shall apply the Stochastic Gradient Descent for an SVM's Hinge Loss
def train(X,Y, epochs, print_cost=True):
    
    # First we initialize our SVM weight's Vector - same dimension as the dimension of X - 3 in this case
    w = np.zeros(len(X[0]))
    
    # Now we define our learning_rate
    learning_rate = 1
    
    # Now we define our total number of epochs
    # epochs = 1000
    
    # Store the miss-classified Data points in "errors" list to plot them later
    errors = list()
    
    # Now we the Interesting Training Part!
    for epoch in range(1, epochs):
        # We set error to zero for each epoch. Don't worry, all errors are added to the list for later analysis! 
        error = 0
        
        # Here we iterate over all indexes and their corresponding data points in X
        for i, _ in enumerate(X):
            
            # If missclassified, update our weights according to misclassified rule
            if (Y[i]*np.dot(X[i], w)) < 1:
                w += learning_rate * ((Y[i] * X[i]) + (-2 * (1/epoch) * w))
                
                # Since we have missclassified, now set the error to the maximum, i.e: 1
                error = 1
                
            # If correctly classified, update our weights according to correctly classified rule
            else:
                w += learning_rate * (-2 * (1/epoch) * w)
            
            if print_cost:
                print("Error at Epoch: ",epoch," is: ",error)
            
            errors.append(error)
            
    # After the training is finished, plot errors
    plt.plot(errors, "|")
    # plt.ylim(0.5, 1.5)
    # plt.axes().set_yticklabel([])
    plt.title("Evolution of Errors by epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Missclassified")
    plt.show()
    
    return w

# Run this bad boi ;)
epochs = int(input("Please enter number of Epochs: "))
w_optim = train(X,y, epochs=epochs)

print("-------------------------------------------")
print("Model has finished training")
print("Code Author: Tanay Mehta\nGithub:/heytanay\nTwitter:/Tanaymehta28")
print("-------------------------------------------")

exit()