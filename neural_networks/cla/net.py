import numpy as np
import random

class neural_net:
    def __init__(self,num_input,next_layer):
        A=np.array([1,1,1,1,1,1,1,1,1,1])
        self.num_input=num_input
        self.next_layer=next_layer
        self.weight=[]
        for i in range(0,10):
            w=(random.randint(0,10)+random.randint(0,5))
            self.weight.append(w)
        print(A,self.weight)      
        a=np.multiply(A,self.weight)
        a=self.weighting(a)
        print(a)


    def weighting(self,weight):
    	i=1
    	a=2
    	print(self.weight[i])
    	if self.weight[i]-a>0:
    		self.weight[i]*=.8
    		self.weighting(self.weight)
    	return self.weight

neural_net(10,1)