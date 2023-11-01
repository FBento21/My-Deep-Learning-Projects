import numpy as np
import matplotlib.pyplot as plt

### DATA GENERATION ###

x = np.linspace(-np.pi, np.pi, 300)

f = lambda x: 0.5*np.sin(2*x)

y = f(x)

### AUXILIARY FUNCTIONS ###

def activation(x):
    return 2*np.arctan(x)/np.pi

def activation_grad(x):
    return 2/(np.pi*(x**2+1))

def cost_function(true, predictions):
    return np.linalg.norm(true - predictions)**2/2

### NEURAL NETWORK ###

class SimpleNN:
    def __init__(self, arch, eta):
        self.arch = arch
        self.eta = eta
        self.weights_matrix = [] # List with the weigths for each layer
        self.biases_matrix = [] # List with the biases for each layer
        self.zz = [] # List with neuron values in each layer
        self.aa = [] # List with neuron activation values in each layer
        self.deltas = [] # List with the neuron's errors in each layer
        self.cost_grad_weights = [] # List of the partial derivatives of the cost_function with respect to the weights
        self.cost_grad_biases = [] # List of the partial derivatives of the cost_function with respect to the biases
        self.layers = len(self.arch)
        for i in range(1, self.layers): #Create random weights and biases
            self.weights_matrix.append(np.random.rand(self.arch[i], self.arch[i-1]))
            self.biases_matrix.append(np.random.rand(self.arch[i], 1))
            self.cost_grad_weights.append(np.zeros((self.arch[i], self.arch[i-1]))) # Initialize the partial derivatives with zeros
            self.cost_grad_biases.append(np.zeros((self.arch[i], 1)))

    def feed_forward(self, x):
        a = x
        self.aa.append(a)
        for i in range(self.layers - 1): # Feed forwared while recording each weight and activation
            z = np.dot(self.weights_matrix[i], a) + self.biases_matrix[i]
            a = activation(z)
            self.zz.append(z)
            self.aa.append(a)

    def backpropagation(self, y):
        delta_L = (self.aa[-1] - y)*activation_grad(self.zz[-1]) # Error of the last layer
        self.deltas.append(delta_L) # Record error of last layer
        for l in range(2, self.layers):
            weigths = self.weights_matrix[-l+1].T
            delta_l = np.dot(weigths, self.deltas[l-2])*activation_grad(self.zz[-l])
            self.deltas.append(delta_l)
        self.deltas.reverse() # [delta_L,..., delta_2] -> [delta_2,..., delta_L]

        #self.cost_grad_biases = np.sum([self.cost_grad_biases, self.deltas], axis=0)
        for l in range(0, self.layers-1):
            self.cost_grad_biases[l] += self.deltas[l]
            self.cost_grad_weights[l] += np.outer(self.aa[l],self.deltas[l]).T # [a^l_1,... a^l_K].([delta^l_1,..., delta^l_J])^T
        self.zz = [] # Re-initialize the neuron's values
        self.aa = [] # Re-initialize the neuron's activations
        self.deltas = [] # Re-initialize the neuron's errors

    def gradient_descent(self, batch_size):
        for l in range(0, self.layers-1):
            self.biases_matrix[l] -= self.eta * self.cost_grad_biases[l]/batch_size
            self.weights_matrix[l] -= self.eta * self.cost_grad_weights[l]/batch_size

    def get_predictions(self, x):
        a = x
        for i in range(self.layers - 1):
            z = np.dot(self.weights_matrix[i], a) + self.biases_matrix[i]
            a = activation(z)
        return a


    def train(self, n_iter, x, y, n_batches):
        for iteration in range(n_iter): #Go through each epoch
            # First shuffle data arrays
            data_size = x.size
            s = np.arange(0, data_size, 1)
            np.random.shuffle(s)
            x_s = x[s]
            y_s = y[s]
            batch_size = data_size // n_batches

            # Then create batches
            x_batches = []
            y_batches = []

            for n in range(n_batches):
                x_batches.append(x_s[n*batch_size: (n+1)*batch_size])
                y_batches.append(y_s[n*batch_size: (n+1)*batch_size])

                if n == n_batches - 1:
                    x_batches.append(x_s[n*batch_size:])
                    y_batches.append(y_s[n*batch_size:])

            # After creating batches start the stochastic gradient descent with mini batches
            print(f'Iteration {iteration+1}')
            for x_batch, y_batch in zip(x_batches, y_batches): # Chose a mini batch
                for xx, yy in zip(x_batch, y_batch): # Iterate through each element in the mini batch
                    self.feed_forward(xx) # Feedforward each xx
                    self.backpropagation(yy) # Backpropagate the xx with yy
                self.gradient_descent(batch_size) # After receiving the mini batch apply the gradient descent

                # Re-initialize the partial derivatives to zero
                self.cost_grad_biases = []
                self.cost_grad_weights = []
                for i in range(1, self.layers):
                    self.cost_grad_weights.append(np.zeros((self.arch[i], self.arch[i-1])))
                    self.cost_grad_biases.append(np.zeros((self.arch[i], 1)))

        predicted = []
        for x_i in x:
            pred =  self.get_predictions(x_i)[0, 0]
            print(pred)
            predicted.append(pred)

        final_error = cost_function(y, predicted)

        print("Error: ", final_error)

        return predicted


arch = [1, 20, 10, 10, 1]
eta = 2


nn = SimpleNN(arch, eta)
predicted = nn.train(100, x, y, 10)

### Final Plot

plt.plot(x, y, label='True')
plt.plot(x, predicted, label='Predicted')

plt.show()


