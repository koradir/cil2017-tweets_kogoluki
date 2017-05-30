"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from statusbar import status_update

#inputMatrix = np.random.rand(50000,200);
#k = 4;

""""def status_update(current, top, label="Progress"):
    workdone = current/top
    print("\r{0:s}: [{1:30s}] {2:.1f}%".format(label,'#' * int(workdone * 30), workdone*100), end="", flush=True)
    if workdone == 1:
        print()"""
		

def kmeans(inputMatrix, k,iterationCount=20):

    numberOfPoints = inputMatrix.shape[0]
    dimensions = inputMatrix.shape[1]
    
    #colors = np.array(['bo', 'go', 'ro','co','mo','yo','ko'])
    
    clusterNumber = np.random.randint(0,k,numberOfPoints)
    
    centers = np.empty([k, dimensions])
    
    randomNumbers = np.random.choice(numberOfPoints, k, replace=False)
    
    for i in range(k):
        centers[i] = inputMatrix[randomNumbers[i]]
                               
    #print(centers)
    
    #plt.figure(1)
    
    """for i in range(numberOfPoints):
        plt.plot(inputMatrix[i][0],inputMatrix[i][1],colors[clusterNumber[i]])
        
    for i in range(k):
        plt.plot(centers[i][0],centers[i][1],'kD')"""
    
    for iteration in range(iterationCount):
        
        error = 0
        
        #--------------------Begin of Cluster Assignment-------------------
        
        for i in range(numberOfPoints):
            if i % 1000 == 0:
                status_update(i+iteration*numberOfPoints,(numberOfPoints-1)*iterationCount-1)
                
            assignPoint = inputMatrix[i]
            bestDistance = np.inf
            for j in range(k):
                candPoint = centers[j]
                distance = np.linalg.norm(assignPoint-candPoint)
                error = error + distance;
                if(distance < bestDistance):
                    bestDistance = distance
                    clusterNumber[i] = j
                                 
        #--------------------End of Cluster Assignment-------------------
        
        #plt.figure(iteration*2+1)
        
        """for i in range(numberOfPoints):
            plt.plot(inputMatrix[i][0],inputMatrix[i][1],colors[clusterNumber[i]])
            
        for i in range(k):
            plt.plot(centers[i][0],centers[i][1],'kD')"""
        
        #-------------------Begin of new mean computation----------------
        
        centers = np.zeros([k, dimensions])
        counts = np.zeros([k, 1])
        
        for i in range(numberOfPoints):
            assignedCenterNumber = clusterNumber[i]
            centers[assignedCenterNumber] = centers[assignedCenterNumber] + inputMatrix[i]
            counts[assignedCenterNumber] = counts[assignedCenterNumber]+1
        
        for j in range(k):
            centers[j] = centers[j] / counts[j];
    
    #plt.figure()
    
    #colours = [clusterNumber[i]+1 for i in range(numberOfPoints)]
    #plt.scatter(inputMatrix[:,0],inputMatrix[:,1],c=colours)

    #for i in range(numberOfPoints):
        #plt.plot(inputMatrix[i][0],inputMatrix[i][1],colors[clusterNumber[i]])
        
    #for i in range(k):
        #plt.plot(centers[i][0],centers[i][1],'kD')
        
    #print(error)
    
    return centers
            
        
 