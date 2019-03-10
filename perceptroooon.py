import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class SingleLayerPerceptron(object):
    def __init__(self, iris, n_epoch, n_fold, learning_rate):
        self.iris = iris
        self.epoch = n_epoch
        self.n_fold = n_fold
        self.learning_rate = learning_rate
        self.weight = [0.5, 0.5, 0.5, 0.5]
        self.bias = 0.5
        self.training_accuracy_list = []
        self.training_error_list = []
        self.val_accuracy_list = []
        self.val_error_list = []

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def get_error(self, act, target):
        return np.power((act-target), 2)

    def prediction(self, act):
        return 1 if act >= 0.5 else 0

    def get_result(self, iris):
        res = 0
        for i in range(len(iris)):
            res += iris[i]*self.weight[i]
        res += self.bias
        return res
  
    def update_weight(self, iris, act):
        for i in range(len(iris)-2):
            d_weight = 2*iris[i]*(iris[4]-act)*(1-act)*act
            self.weight[i] = self.weight[i] + (self.learning_rate)

    def update_bias(self, iris, act):
        d_bias = 2*(iris[4]-act)*(1-act)*act
        self.bias = self.bias + (self.learning_rate*d_bias)
 
    def cross_validation_split_train(self):
        training_error = 0
        val_error = 0
        training_accuracy = 0
        val_accuracy = 0
        counter1 = 0
        counter2 = 0

        dataset_split = []
        training_data = []

        for i in range(self.n_fold):
            dataset_split.append(self.iris[20*i:(20*i)+20])

        for i in range(self.n_fold):
            val_data = dataset_split[i]

            for j in [x for x in range(self.n_fold) if x != i]:
                training_data.extend(dataset_split[j])

            for iris in training_data:
                result_ = self.get_result(iris[:4])
                activation_ = self.sigmoid(result_)
                prediction_ = self.prediction(activation_)
                err1 = self.get_error(iris[4], activation_)
                
                self.update_weight(iris, activation_)
                self.update_bias(iris, activation_)

                if(prediction_ == iris[4]):
                    training_accuracy += 1
           
                training_error += err1
                counter1 += 1

            for iris in val_data:
                result_ = self.get_result(iris[:4])
                activation_ = self.sigmoid(result_)
                prediction_ = self.prediction(activation_)
                err2 = self.get_error(iris[4], activation_)

                if(prediction_ == iris[4]):
                    val_accuracy += 1
            
                val_error += err2
                counter2 += 1

        self.training_accuracy_list.append(training_accuracy/counter1)
        self.training_error_list.append(training_error/counter1)
        self.val_accuracy_list.append(val_accuracy/counter2)
        self.val_error_list.append(val_error/counter2)
  
    def do_train(self):
        for i in range(self.epoch):
            self.cross_validation_split_train()

    def plot_error(self):
        plt.plot(np.arange(self.epoch), self.training_error_list, color='blue', label='Training Error')
        plt.plot(np.arange(self.epoch), self.val_error_list, color='green', label='Validation Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Grafik Error')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        plt.plot(np.arange(self.epoch), self.training_accuracy_list, color='red', label='Training Accuracy')
        plt.plot(np.arange(self.epoch), self.val_accuracy_list, color='green', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Grafik Akurasi')
        plt.legend()
        plt.show()  

def main():
    iris = pd.read_csv('iris-target.csv').values

    # slp1 = SingleLayerPerceptron(iris, 300, 5, 0.1)
    # slp1.do_train()
    # slp1.plot_error()
    # slp1.plot_accuracy()
    slp2 = SingleLayerPerceptron(iris, 300, 5, 0.8)
    slp2.do_train()
    slp2.plot_error()
    slp2.plot_accuracy()

if __name__ == "__main__":
    main()