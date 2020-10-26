from numpy import *
import csv

class NeuralNet(object): 
    def __init__(self): 
        # Generate random numbers 
        random.seed(1) 
  
        # Assign random weights to a 6 x 1 matrix, 
        self.synaptic_weights = 2 * random.random((6, 1)) - 1 
        #[0.2,0.4,0.1,0.1,0.05]
        #2 * random.random((5, 1)) - 1 
  
    # Сигмоида 
    def __sigmoid(self, x): 
        return 1 / (1 + exp(-x)) 
  


    
    # градиент сигмоиды
    def __sigmoid_derivative(self, x): 
        return x * (1 - x) 
  
    # обучаем сеть и переназначаем веса каждый раз
    def train(self, inputs, outputs, training_iterations): 
        for i in range(training_iterations): 
  
            # передаем набор данных в сеть
            output = self.learn(inputs) 
  
            # считаем ошибки
            error = outputs - output 
  
            # настраиваем веса по факторам
            factor = dot(inputs.T, error * self.__sigmoid_derivative(output)) 
            self.synaptic_weights += factor 
  
    # обучаем 
    def learn(self, inputs): 
        return self.__sigmoid(dot(inputs, self.synaptic_weights)) 
  
if __name__ == "__main__": 
  
    with open('./inputfile2.csv', newline='') as f: # исходные данные для обучения
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        data = list(reader)
        for x in data:
            x = [float(i) for i in x] 
        

    with open('./res.csv', newline='') as f: # установленные диагнозы данные
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        res = list(reader)
        for x in res:
            x = [float(i) for i in x] 
       

    with open('./sample2.csv', newline='') as f: # тестовые данные
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        sample = list(reader)
        
        for x in sample:
            x = [float(i) for i in x]

    #инизиализируем сеть 
    neural_network = NeuralNet() 
  
    # набор данных для обучения
    inputs = array(data) 
    outputs = array(res).T  #результаты
  
    # обучаем сеть 
    neural_network.train(inputs, outputs, 10000) 
  
    # отображаем вывод
    print (neural_network.learn(array(sample)) )