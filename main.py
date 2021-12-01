from pandas import read_csv
import numpy as np
# Load dataset
dataset = read_csv("/home/galguene/Documents/Code/py/smashpredict/smash.csv", names=['name', 'party', 'swordie', 'anime', 'gotsmash'])
array = dataset.values
inp = array[:,1:4]
out = [array[:,4]]
print(out)

class NeuralNetwork():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights = self.synaptic_weights + adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights).astype(float))
        return output

if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array(inp)

    training_outputs = np.array(out).T

    #training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

print(0,0,0,":",neural_network.think(np.array([0,0,0])))
print(0,0,1,":",neural_network.think(np.array([0,0,1])))
print(0,1,0,":",neural_network.think(np.array([0,1,0])))
print(0,1,1,":",neural_network.think(np.array([0,1,1])))
print(1,0,0,":",neural_network.think(np.array([1,0,0])))
print(1,0,1,":",neural_network.think(np.array([1,0,1])))
print(1,1,0,":",neural_network.think(np.array([1,1,0])))
print(1,1,1,":",neural_network.think(np.array([1,1,1])))	
