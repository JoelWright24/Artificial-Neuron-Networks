# Gender, Height (cm), Weight (kg), Generation.
import random
import numpy as np

class Reproduction:
    def forward(self, Father, Mother):
        ChildM1=[Father[0], random.uniform(random.uniform(0.9, 1), random.uniform(1, 1.1))*(1*(Father[1])+0.001*(Mother[1])), random.uniform((random.uniform(0.95, 1)),(random.uniform(1, 1.05)))*(1*(Father[2])+0.001*(Mother[2])) + random.uniform(-5,5), Father[3]+1]
        ChildM2=[Father[0],random.uniform(random.uniform(0.925, 1), random.uniform(1, 1.075))*(1*(Father[1])+0.001*(Mother[1])), random.uniform((random.uniform(0.9, 1)),(random.uniform(1, 1.1)))*(1*(Father[2])+0.001*(Mother[2])) + random.uniform(-1,1), Father[3]+1]
        ChildF1=[Mother[0], random.uniform(random.uniform(0.9, 1), random.uniform(1, 1.1))*(0.001*(Father[1])+1*(Mother[1])), random.uniform((random.uniform(0.95, 1)),(random.uniform(1, 1.05)))*(0.001*(Father[2])+1*(Mother[2])) + random.uniform(-2.5,2.5), Father[3]+1]
        ChildF2=[Mother[0], random.uniform(random.uniform(0.95, 1), random.uniform(1, 1.05))*(0.001*(Father[1])+1*(Mother[1])), random.uniform((random.uniform(0.8, 1)),(random.uniform(1, 1.2)))*(0.001*(Father[2])+1*(Mother[2])) + random.uniform(-5,5), Father[3]+1]

        self.Children=[ChildM1, ChildM2, ChildF1, ChildF2]
        
       


class Simultion:
    def Like_Rabbits(self, Father, Mother, Number_of_Generations, Existing_Population):
        
        if (Existing_Population!=[]):
            Total_Population=[]
            Total_Population.append(Existing_Population)
            Total_Population=np.array(Total_Population)

        counter=0
        #initialize Damen, Herren and the child lists
        Damen=[]
        Damen.append(Mother)
        Herren=[]
        Herren.append(Father)      
        childList_M=[]
        childList_F=[]
        while (counter<Number_of_Generations):
                    
            for jj in range (len(Damen)):
                Father=Herren[jj]
                Mother=Damen[jj]      
                Next_Litter=Reproduction()
                Next_Litter.forward(Father, Mother)
                childList=Next_Litter.Children

                Male_offspring=[]
                Female_offspring=[]
                for i in range (len(childList)):
                    child=childList[i]
                    if (child[0] == -1):
                        Female_offspring.append(child)
                    else:
                        Male_offspring.append(child)
                if jj==0:
                    childList_M=np.vstack([Male_offspring])
                    childList_F=np.vstack([Female_offspring])
                else:
                    childList_M=np.vstack([childList_M, Male_offspring])
                    childList_F=np.vstack([childList_F, Female_offspring])    
            if counter==0:
                Past_Damen=np.vstack(Damen)
                Past_Herren=np.vstack(Herren)
            else:
                Past_Damen=np.vstack([Past_Damen, Damen])
                Past_Herren=np.vstack([Past_Herren, Herren])
            Damen=childList_F
            Herren=childList_M
            counter=counter+1         
            #now to shuffle
            np.random.shuffle(np.array(Damen))
            np.random.shuffle(np.array(Herren))
        #What to return?
        
        Total_Population=np.vstack([Past_Damen, Past_Herren, Damen, Herren])
        np.savetxt("HeightAndWeight.csv", Total_Population, delimiter=",")
        self.output=Total_Population

'''

### Looking into the data: 

Father=[1, 180, 100, 0]
Mother=[-1, 160, 60, 0]
Existing_Population=[]

Example_8=Simultion()
Example_8.Like_Rabbits(Father, Mother, 8, Existing_Population)
Output_Data=Example_8.output
print (Example_8.output)


Mean_Weight_M=[]
Mean_Weight_F=[]
Mean_Height_M=[]
Mean_Height_F=[]
for jj in range (9):
    Mean_Weight_M_CL=[]
    Mean_Weight_F_CL=[]
    Mean_Height_M_CL=[]
    Mean_Height_F_CL=[]
    for ii in range (len(Output_Data)):
        solution=Output_Data[ii,:]
        if solution[3]==jj:
            if solution[0]==-1:
                Mean_Weight_F_CL.append(solution[2])
                Mean_Height_F_CL.append(solution[1])
            else:
                Mean_Weight_M_CL.append(solution[2])
                Mean_Height_M_CL.append(solution[1])

        if ii==(len(Output_Data)-1):
            Mean_Weight_M.append(np.mean(Mean_Weight_M_CL))
            Mean_Weight_F.append(np.mean(Mean_Weight_F_CL))
            Mean_Height_M.append(np.mean(Mean_Height_M_CL))
            Mean_Height_F.append(np.mean(Mean_Height_F_CL))                
print (Mean_Height_M)
print (Mean_Height_F)
print (Mean_Weight_M)
print (Mean_Weight_F)

'''



