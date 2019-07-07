import numpy as np

class NeuralNet:
    def __init__(self,num_hidden,num_neuron,learning_rate):
        self.num_input=7000                  #took the 7000 samples given for training
        self.dimension=784        
        self.num_out=10                      #output is 10 classes
        self.layers=[]
        self.weights=[]
        self.bias=[]
        self.num_hidden=num_hidden
        self.num_neuron=num_neuron
        self.learning_rate=learning_rate
        self.X_train=np.zeros((7000,784))     #change 7000 to 60,000 if you want to use the full MNIST dataset
        self.Y_train=np.zeros(7000)           #change 7000 to 60,000 if you want to use the full MNIST dataset
        self.d_weights=[]
        self.d_bias=[]
        self.d_layers=[]
        self.Y_train_hot=np.zeros((7000,10)) #change 7000 to 60,000 if you want to use the full MNIST dataset
    
    def relu(self,x):                         #relu for nonlinearity
        return np.maximum(0,x)
    
    def get_data(self):                       # getting the data
        Data= np.genfromtxt(r'mnist_train.csv', delimiter=',')
        self.Y_train=Data[:,0]
        self.X_train=Data[:,1:]/255
        self.Y_train=self.Y_train.astype(int)
        self.Y_train_hot[range(self.num_input),self.Y_train]=1          #one hot representation of training examples
        
    def weight_inialization(self):           #initialising the weights for each layer performance can change with values of alpha
        self.get_data()
        alpha=.1
        W=alpha*np.random.randn(self.dimension,self.num_neuron)
        self.weights.append(W)
        b=np.zeros((1,self.num_neuron))
        self.bias.append(b)
        for i in range(1,self.num_hidden):
            W=alpha*np.random.randn(self.num_neuron,self.num_neuron)
            self.weights.append(W)
            b=np.zeros((1,self.num_neuron))
            self.bias.append(b)
        W=alpha*np.random.randn(self.num_neuron,self.num_out)
        self.weights.append(W)
        b=np.zeros((1,self.num_out))
        self.bias.append(b)
                           
    def forward_pass(self):                 #normal forward pass
        j=0
        self.layers=[]
        self.layers.append(np.dot(self.X_train,self.weights[j])+self.bias[j])
        for i in range(1,self.num_hidden):
            j=j+1
            self.layers.append(self.relu((np.dot(self.layers[j-1],self.weights[j])+self.bias[j])))
        x=np.exp((np.dot(self.layers[j],self.weights[j+1]))+self.bias[j+1])
        x=x/np.sum(x,axis=1,keepdims=True)
        self.layers.append(x)
        
    def back_pass(self):                   #backprop 
        j=0
        k=self.num_hidden
        self.d_layers=[]
        self.d_weights=[]
        self.d_bias=[]                                                           #first layer and last layer backprop have to explicitly written
        self.d_layers.append((self.layers[k]-self.Y_train_hot)/self.num_input)
        self.d_weights.append(np.dot((self.layers[k-1]).T,self.d_layers[j]))
        self.d_bias.append(np.sum(self.d_layers[j],axis=0,keepdims=True))
        for i in range(1,self.num_hidden):
            j=j+1
            self.d_layers.append(np.dot(self.d_layers[j-1],(self.weights[k]).T))
            W=self.d_layers[j]
            W[self.layers[k-1]<=0]=0
            self.d_layers[j]=W
            k=k-1
            self.d_weights.append(np.dot((self.layers[k-1]).T,self.d_layers[j]))
            self.d_bias.append(np.sum(self.d_layers[j],axis=0,keepdims=True))
        self.d_layers.append(np.dot(self.d_layers[j],(self.weights[k]).T))
        self.d_weights.append(np.dot((self.X_train).T,self.d_layers[j+1]))
        self.d_bias.append(np.sum(self.d_layers[j+1],axis=0,keepdims=True))
        
    def weight_updation(self):      #updating the weight
        k=self.num_hidden
        for i in range(self.num_hidden):
            self.weights[i]+=-self.learning_rate*self.d_weights[k]
            self.bias[i]+=-self.learning_rate*self.d_bias[k]
            k=k-1
            
    def error(self):
        loss=0.5*np.sum(np.square((self.layers[self.num_hidden]-self.Y_train_hot)))/7000
        return (loss)
    
    def accuracy(self):
        count=0
        predicted=np.argmax(self.layers[self.num_hidden],axis=1)
        for i in range(self.num_input):
            if(predicted[i]==self.Y_train[i]):
                count+=1
        count=count/self.num_input
        print(count)
        
    def train(self):
        epoch=1000           # change the epochs to see the variation
        self.weight_inialization()
        for i in range(epoch):
            self.forward_pass()
            self.back_pass()
            self.weight_updation()
            #print(self.error())
            
    def test(self):
        test_data= np.genfromtxt(r'mnist_test.csv', delimiter=',')
        test_data=test_data/255
        j=0
        self.layers=[]
        self.layers.append(np.dot(test_data,self.weights[j])+self.bias[j])
        for i in range(1,self.num_hidden):
            j=j+1
            self.layers.append(self.relu((np.dot(self.layers[j-1],self.weights[j])+self.bias[j])))
        x=np.exp((np.dot(self.layers[j],self.weights[j+1]))+self.bias[j+1])
        x=x/np.sum(x,axis=1,keepdims=True)
        self.layers.append(x)
        predicted=np.argmax(self.layers[self.num_hidden],axis=1)
        predicted=predicted.astype(int)
        np.savetxt("test_result.csv", predicted, fmt='%10.0f', delimiter='\t')

Net1=NeuralNet(2,300,.1) #declare as NeuralNet(number of hidden layers,number of neurons in hidden layer,learning rate)
Net1.train()
Net1.accuracy()          #training set accuracy
Net1.test()
