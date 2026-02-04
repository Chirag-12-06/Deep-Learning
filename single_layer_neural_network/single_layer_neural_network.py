import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ETA = 0.001
epochs = 1000

class single_layer_neural_network:
    def __init__(self,input_layer,output_layer):
        self.lr= ETA
        self.epochs=epochs
        self.weights=np.random.randn(input_layer, output_layer)
        self.output_layer=output_layer
        self.input_layer=input_layer
        self.E=np.array([])
        self.w_list=[]

    def sigmoid(self,x):
        """Acticvation function: Sigmoid"""
        return 1/(1+np.exp(-x))

    def feedforward(self,X):
        """Forward propagation through the network"""
        linear_output = np.dot(X, self.weights)
        activated_output = self.sigmoid(linear_output)
        return activated_output

    def error(self,h,y):
        """Logarithmic error calculation"""
        error = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        self.E = np.append(self.E, error)

    def backprop(self,X,y,h):
        """Backpropagation algorithm to update weights"""
        =np.dot(X.T,(h-y))/self.output_layer

        self.weights=self.weights - self.lr*self.delta_e_w

    def predict(self,X):
        pred=self.feedforward(X)
        return pred

    def classify(self,y):
        return self.predict(y).round()

    def train(self,X,y):
        for epoch in range(self.epochs):
            h=self.feedforward(X)
            self.backprop(X,y,h)
            # Store error every 10 epochs to reduce overhead
            if epoch % 10 == 0:
                self.error(h,y)
    
    def plot(self):
        """plotting error vs epochs"""
        plt.plot(self.E)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Training Error over Epochs')
        plt.show()



df=pd.read_csv("single_layer_neural_network/glass.data")

# Create binary classification: Window (1-4) vs Non-Window (5-7)
df['Window'] = df['Type'].map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})

# iv = ['Al']
iv = ['RI','Mg','Al','K','Ca']
x = df[iv].values
y = df['Window'].values
input_layer = x.shape[1]
output_layer = 1

# Initiate Single Perceptron NN
SPNN = single_layer_neural_network(input_layer, output_layer)

SPNN.train(x, y)

SPNN.plot()

pred = SPNN.predict(x)

pred2 = SPNN.classify(x)

print("Minimum Error achieved:", min(SPNN.E))

print("Weights:", SPNN.weights)