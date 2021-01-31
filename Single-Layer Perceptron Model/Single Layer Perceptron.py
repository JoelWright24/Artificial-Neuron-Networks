# Have a look at single layer perseptron example
# Artificial Neuro Network for Calssifiaction
# inputs --> Weights --> Summing Junction + bias --> Activation Function --> Results

import numpy as np
import matplotlib.pyplot as plt

# I am not doing the updating of the weights in the most professional way, however, it is functional at the moment.
# Once more has been learned regarding back propigation, this will be updated

class slp:
    def __init__(self, inputs, weights, bias):
        self.bias=bias
        self.inputs=inputs
        self.weights=weights
        
    def forward(self):
        
        self.output=np.dot(self.weights, self.inputs) + bias

class Delta_Rule:
    def forward(self, allData, Inputs, original_weights, eta, desired):
        #delta rule =>
        constant1 = desired - Inputs
        constant2 = eta*constant1
        self.new_weights = np.add(original_weights, Inputs*constant2)
        Nweights=self.new_weights
        # Require test
        def Accuracy_Review(allData, original_weights, bias, eta):
            Accuracy_points=0
            weights=original_weights

            for ii in range (len(allData)):
                inputs=[allData[ii,1],allData[ii,2]]
                solution = slp(inputs, weights, bias)
                solution.forward()
                Activation=Activation_Binary()
                    
                if Activation.forward(solution.output)==allData[ii,0]:
                    Accuracy_points=Accuracy_points+1
            Accuracy_percentage=Accuracy_points/len(allData)
            return (Accuracy_percentage)

        def Accuracy_test(allData, original_weights, Nweights, bias, eta):
            test_previous=Accuracy_Review(allData, original_weights, bias, eta)
            test_new=Accuracy_Review(allData, Nweights, bias, eta)
            better_weighting=max(test_new, test_previous)
            #don't want to print the better weighting, we want to print the original or new weights
            if better_weighting==test_previous:
                return_weight=original_weights
            else:
                return_weight=Nweights
            return (return_weight)
        best_weight=Accuracy_test(allData, original_weights, Nweights, bias, eta)
        return (best_weight)


class Activation_Linier:
    def forward(self, Input):
        #exp_values = np.exp(Input)
        #output=exp_values
        output = Input
        return (output)

class Activation_Binary:
    def forward(self, Input):
        if Input>0:
            output=1
        else:
            output=-1 
        return (output)        

class Single_Pass:
    
    def forward(self, original_weights, bias, eta, height, mass, gender):
        # Call the required OOP classes:
        First_Pass=slp([height, mass], original_weights, bias)
        Activation=Activation_Binary()
        Update=Delta_Rule()        

        #insert the data and call the functions
        First_Pass.forward()
        Activation.forward(First_Pass.output)
        Update.forward(allData, Activation.forward(First_Pass.output), original_weights, eta, gender)
        self.output=Update.new_weights



import random

def data_sort(FileName):
    data=np.genfromtxt(FileName, delimiter=',', dtype=None, encoding=None)
    genderList=[]
    weightList=[]
    heightList=[]
    for ii in range (len(data)):
        person=data[ii]
        genderList.append(person[0])
        heightList.append(person[1]*2.54) # conversion inches to cm
        weightList.append(person[2]/2.205) # conversion lbs to kg
        
    
    genders=np.array(genderList)
    genders_binary=np.zeros(len(genderList))
    for jj in range (len(genders)):
        if genders[jj]=='Male':
            genders_binary[jj]=1.0 # >0 is Male
        else:
            genders_binary[jj]=-1.0 # <0 is Female
    
    #how to shuffle 
    allData=np.array((genders_binary, heightList, weightList))
    allData=np.transpose(allData)
    np.random.shuffle(allData) #Shuffle the data
    return (allData)

#now syphen off the different info streams: input and desired
#define desired_array and input_array
allData=data_sort('weight-height.csv')
# First column of new data shows whether Male=1 or Female=-1
# Second column is height
# Third column is weight

#Randomly sample alldata
def Random_Sample (allData, Number_of_Samples):
    randomlist = []
    counter=0
    while counter<Number_of_Samples:
        n_index = random.randint(1,len(allData))
        randomlist.append(allData[n_index,:])
        counter=counter+1
    return (np.array(randomlist))

def Accuracy_Review(allData, original_weights, bias, eta):
    Accuracy_points=0
    weights=original_weights

    for ii in range (len(allData)):
        inputs=[allData[ii,1],allData[ii,2]]
        solution = slp(inputs, weights, bias)
        solution.forward()
        Activation=Activation_Binary()
            
        if Activation.forward(solution.output)==allData[ii,0]:
            Accuracy_points=Accuracy_points+1
    Accuracy_percentage=Accuracy_points/len(allData)
    return (Accuracy_percentage)

def Accuracy_test(allData, original_weights, new_weights, bias, eta):
    test_previous=Accuracy_Review(allData, original_weights, bias, eta)
    test_new=Accuracy_Review(allData, new_weights, bias, eta)
    better_weighting=max(test_new, test_previous)
    #don't want to print the better weighting, we want to print the original or new weights
    if better_weighting==test_previous:
        return (original_weights)
    else:
        return (new_weights)    

def Full_Pass(SampledData, original_weights, bias, eta):
    new_weights=original_weights # required for the first pass
    for ii in range (len(SampledData)):
        current_row=SampledData[ii,:]
        current_height=current_row[1]
        current_mass=current_row[2]
        current_gender=current_row[0]
        # Place info
        NochMal=Single_Pass()
        NochMal.forward(new_weights, bias, eta, current_height, current_mass, current_gender) #may require allData at start
        new_weights=NochMal.output 
    # Now run an accuracy test
    better_weights=Accuracy_test(SampledData, original_weights, new_weights, bias, eta)
        
    return (better_weights)

def Multiple_Pass(epochs,SampledData, original_weights, bias, eta):
    output_weights=original_weights
    counter=1
    while counter<epochs:
        output_weights=Full_Pass(SampledData, output_weights, bias, eta)
        counter=counter+1

    return (output_weights)    

#original_weights=np.array([-39, 90]) # These may be randomly chosen at the start
def Select_Initial_Weights(Choice_Percentage):
    original_weights=np.array([random.uniform(-35, -25), random.uniform(92, 98)])
    weights_list=[]
    min_list=[]
    counter=0
    while (counter<50): #gives 50 options
        random_weights=[random.uniform(-35, -25), random.uniform(92, 98)]
        weights_list.append(random_weights)
        min_list.append(abs(Accuracy_Review(allData, random_weights, bias, eta) - (Choice_Percentage/100))) #closest value to 60%
        counter=counter+1
    required_index=np.argmin(min_list)
    original_weights=weights_list[required_index]
    return (original_weights)
#Idea of implimenting a selection for the best random starter. 
#  
bias=70
eta=0.1
Number_of_Samples=50
original_weights=Select_Initial_Weights(53) #The reason why I have selected a starting accuracy percentage of 53% is due to a lack of local minimums around here.

Initial_Accuracy=Accuracy_Review(allData, original_weights, bias, eta)

Second_Weights=Full_Pass(Random_Sample(allData, 20), original_weights, bias, eta)
Next_Accuracy=Accuracy_Review(allData, Second_Weights, bias, eta)

epochs=10
Third_Weights=Multiple_Pass(epochs, Random_Sample(allData, 10), Second_Weights, bias, eta)
Final_Accuracy=Accuracy_Review(allData, Third_Weights, bias, eta)

#Printing Statements:
print("The Initial Accuracy Percentage of weights ",original_weights,"with the initial weights is", Initial_Accuracy*100,"%")
print("The Accuracy Percentage of weights", Second_Weights," with single pass editing the weights is", Next_Accuracy*100,"%" )
print("And the Accuracy Percentage passing through", epochs,"gives the weights", Third_Weights,"and accuracy", Final_Accuracy*100,"%" )


# Older Work:
def plot_data(original_weights,allData): #Plotting the shuffled data
    samplingRate=100
    plt.figure(1)
    for ii in range (int(len(allData)/samplingRate)):
    
        if allData[ii,0]==1: 
            plt.scatter(allData[ii*samplingRate,1],allData[ii*samplingRate,2], color='blue', label="Male")
        else:
            plt.scatter(allData[ii*samplingRate,1],allData[ii*samplingRate,2], color='red', label="Female")

    #plotting line
    m= original_weights[1]/original_weights[0] #gradient
    x1=np.mean(allData[:,1])
    y1=np.mean(allData[:,2])
    

    example_heights=(140, 150, 160, 170, 180, 190)
    example_weights=[]
    for ii in range (len(example_heights)):
        #equation y=m*(x-x1)+y1
        example_weights.append(m*(example_heights[ii]-x1)+y1)

    plt.plot(example_heights,example_weights, color='green')

    plt.xlabel("Height [cm]")
    plt.ylabel("Weight [kg]")
    plt.grid()
    plt.show()


#can set to loop to maximise the chance of being right and also can plot to see the data and see the line change position :) 

#Printing Gender Based on The dataset, and a provided weight and Height
def Gender_Printer(Name, Height, Mass):
    print("### Welcome to GENDER PRINTER ###")
    print("Please Note: Height should be entered in [cm] and Mass in [kg]")
    First_step=slp([Height, Mass], Third_Weights, bias)
    First_step.forward()
    Names_activation=Activation_Binary()
    Gender_predicted=Names_activation.forward(First_step.output)
    if Gender_predicted==1:
        Gender_script="Male"
    else:
        Gender_script="Female"    

    return(print("According to the dataset, the Height and Weight provided and training of this SLP, GENDER PRINTER says",Name,"is", Gender_script, ". Hopefully that is correct :)"))

Gender_Printer("Joel", 178, 68)
Gender_Printer("Velobte", 170, 53)

plot_data(Third_Weights,allData)