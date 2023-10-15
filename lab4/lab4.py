import csv
import numpy as np

def load_data(path):

    X = []
    y = []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            X.append([float(i) for i in row[:-1]]) 
            y.append(float(row[-1]))  
            
    return X, y

class NeuralNetwork:
    
    def __init__(self, input_dim, h_layers, output_dim, X,Y):
        self.input_dim = input_dim
        self.hidden_layers = h_layers
        self.output_dim = output_dim
        self.input_data = np.array(X)
        self.Y = np.array(Y)
        self.initialize_weights()
        

    def initialize_weights(self):
        self.weights = []
        self.bias_weights = []
        prev_dim = self.input_dim

        for layer_dim in self.hidden_layers:
            self.weights.append(np.random.normal(0, 0.01, (prev_dim, layer_dim)))
            self.bias_weights.append(np.random.normal(0, 0.01, (layer_dim,)))
            prev_dim = layer_dim

        self.weights.append(np.random.normal(0, 0.01, (prev_dim, self.output_dim)))
        self.bias_weights.append(np.random.normal(0, 0.01, (self.output_dim,)))

    def train(self):

        predicted_values = []
        for i,input in enumerate(self.input_data):
            out = self.calculate_output(input)
            predicted_values.append(out)
        
        return self.calculate_mse(predicted_values)
        
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def calculate_output(self, input):
        for i in range(len(self.hidden_layers)):
            input = self.sigmoid(np.dot(self.weights[i].T, input) + self.bias_weights[i])
        input = np.dot(self.weights[-1].T, input) + self.bias_weights[-1]
        return input

    def get_weights(self):
        return self.weights
    
    def get_bias_weights(self):
        return self.bias_weights
    
    def set_weights(self, weights):
        self.weights = weights
        
    def set_bias_weights(self, bias_weights):
        self.bias_weights = bias_weights
        
    def calculate_mse(self,predicted_values):
        N= len(predicted_values)
        mse = 0
        
        for x,y in zip(predicted_values, self.Y):
            mse += (x-y)**2
        
        return (mse/N).item()
    
class GeneticAlgorithm:
    
    def __init__(self, popsize, elitism, p, K, iter,train_x, train_y, test_x, test_y, h_layers):
        self.popsize = popsize
        self.elitism = elitism
        self.p = p
        self.K = K
        self.iter = iter
        self.input_dim = len(train_x[0])
        self.hidden_layers = h_layers
        self.output_dim = 1
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        
    def initialize_population(self):
        self.population = []
        
        for i in range(self.popsize):
            nn = NeuralNetwork(self.input_dim, self.hidden_layers, self.output_dim, self.train_x,self.train_y)
            fit = nn.train()
            self.population.append((nn,fit))
  
    
    def calculate_mse_gg(self,predicted_values, target):
        N= len(predicted_values)
        mse = 0
        
        for x,y in zip(predicted_values, target):
            mse += (x-y)**2
        
        return (mse/N).item()
    
    def start(self):
        
        self.initialize_population()
        
        self.evaluate()
        
        for i in range(self.iter):
            new_population = []

            if (i+1) % 2000 == 0:
                print(f'[Train error @{i+1}]: {self.population[0][1]}')
            
            for i in range(self.popsize):
                r1 = self.select_parent()
                r2 = self.select_parent()
                
                d = self.crossover(r1,r2)
                
                d = self.mutate(d)
                
                new_population.append((d,d.train()))
            
            new_population = self.elitism_selection(new_population)
            self.population = new_population
            
            self.evaluate()
        print(f'[Test error]: {self.calculate_mse_gg([self.population[0][0].calculate_output(i) for i in np.array(self.test_x)], self.test_y)}')
       
    def evaluate(self):
        self.population.sort(key = lambda x: x[1])
        
        
    def elitism_selection(self, new_population):
        new_population.sort(key = lambda x: x[1])
        self.population.sort(key = lambda x: x[1])
        new_population = new_population[:-self.elitism]
        
        for i in range(self.elitism):
            new_population.append(self.population[i])
            
        return new_population
    
    def select_parent(self):
        self.population.sort(key = lambda x: x[1])
        
        totalsum = sum(1/mse for nn,mse in self.population)
        
        rand = np.random.uniform(0,totalsum)
        sum_of_roulette = 0
        for nn,mse in self.population:
            sum_of_roulette += 1/mse
            if sum_of_roulette > rand:
                return nn
    
    def crossover(self, nn1, nn2):
        d = NeuralNetwork(self.input_dim, self.hidden_layers, self.output_dim, self.train_x, self.train_y)

        w_par1 = nn1.get_weights()
        w_par2 = nn2.get_weights()
        
        w_child = []
        b_child = []
        for i in range(len(w_par1)):
            w_child.append((w_par1[i] + w_par2[i])/2)
            
        for i in range(len(nn1.get_bias_weights())):
            b_child.append((nn1.get_bias_weights()[i] + nn2.get_bias_weights()[i])/2)
            
        d.set_weights(w_child)
        d.set_bias_weights(b_child)
        
        return d
    
    def mutate(self,nn1):
        
        w = nn1.get_weights()
        b = nn1.get_bias_weights()
        
        for i in range(len(w)):
            if np.random.uniform(0,1) < self.p:
                w[i] += np.random.normal(0, self.K, w[i].shape)
                
        for i in range(len(b)):
            if np.random.uniform(0,1) < self.p:
                b[i] += np.random.normal(0, self.K, b[i].shape)
        
        nn1.set_weights(w)
        nn1.set_bias_weights(b)
        
        return nn1
    
import sys
        
def main():
    
    train_data = sys.argv[2]
    test_data = sys.argv[4]
    nn = sys.argv[6]
    popsize = sys.argv[8]
    elitism = sys.argv[10]
    p = sys.argv[12]
    K = sys.argv[14]
    iter = sys.argv[16]
    
    if nn == '5s':
        h_layers = [5]
    elif nn == '20s':
        h_layers = [20]
    else:
        h_layers = [5,5]
        
    train_x, train_y = load_data(train_data)
    test_x, test_y = load_data(test_data)
    
    gg = GeneticAlgorithm(int(popsize), int(elitism), float(p), float(K), int(iter), train_x, train_y, test_x, test_y, h_layers)
    gg.start()

if __name__ == '__main__':
    main()
