# Date: Oct. 17th, 2022
# Reference: IJCAI-19 Paper ID: 6035
# Balanced Clustering: A Uniform Model and Fast Algorithm


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.linalg import eig
from matplotlib.ticker import MaxNLocator
import time




# download iris data and read it into a dataframe
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
# Wine: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
data = df.loc[:,['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
data = data.values

N = len(data)   # Number of data points
M = len(data[0]) # Dimension of data
K = 3           # Number of attributes
N_iter = 100     # Maximum number of iterations

T = np.zeros((K*N,N_iter+1)) # Cluster labels
C = np.zeros((K,M,N_iter+1)) # Cluster center
D = np.zeros((N,K,N_iter+1)) # Distance
l = 1e-1 # Constraint coefficient
# For Iris, l_opt ~1e-1
n_k = np.zeros((K,N_iter+1)) # Cluster size
obj_value = np.zeros((1,N_iter+1)) # Value of objective function


# Perform PCA on original data to visualize the results
mean_data = sum(data)/float(len(data))
cov_data = np.cov(np.transpose(data))
l_data, U_data = eig(cov_data)
L_data = np.diag(l_data)
M_data = 2       # Dimension of data after PCA
Z_data = np.zeros((N,M_data))
for i in range(M_data):
    for j in range(N):
        Z_data[j,i] = np.dot((data[j,:]-mean_data),U_data[:,i])
n = int(N/K)
for i in range(0,n):        
    plt.scatter(Z_data[i,0],Z_data[i,1],c='red')
    plt.scatter(Z_data[i+n,0],Z_data[i+n,1],c='green')
    plt.scatter(Z_data[i+2*n,0],Z_data[i+2*n,1],c='blue')
    plt.xlabel("1st significant component")
    plt.ylabel("2nd significant component")
    plt.title("Classification Labels")
plt.show()


def dis(C):    # Sum of Square distance
    D = np.zeros((N,K))
    for i in range (N):
        for j in range (K):
            D[i,j] = np.sum(np.square(data[i]-C[j]))
    return D

def count(T):     # Count number of instances in each cluster
    n_k = np.zeros(K)
    for i in range (N):
        for j in range (K):
            if T[i*K+j] == 1:
                n_k[j] += 1
    return n_k
        
def reg(T):       # Regularization function
    n_k = count(T)
    return np.sum(np.square(n_k))

def obj(C,T):  # Objective function
    sum = l * reg(T[:])
    D = dis(C)
    for i in range (N):
        for j in range (K):
            sum += T[i*K+j]*D[i,j]
    return sum



# Initializataion n_iter = 0
# Randomly assign K cluster centers
C_index = np.random.randint(0,N-1,K)
for i in range (K):
    C[i,:,0] = data[C_index[i],:]
# Assign the data points to the closest cluster center
# Without considering the balanced constraints
D[:,:,0] = dis(C[:,:,0])
for i in range (N):
    for j in range (K):
        if j == np.argmin(D[i,:,0]):
            T[i*K+j,0] = 1  
n_k[:,0] = count(T[:,0])
obj_value[:,0] = obj(C[:,:,0],T[:,0])
'''
# Plot cluster result: Iris only
for i in range(N):
        if (T[i*K+0,0] == 1):
            plt.scatter(Z_data[i,0],Z_data[i,1],c='red')
        elif (T[i*K+1,0] == 1):
            plt.scatter(Z_data[i,0],Z_data[i,1],c='green')
        elif (T[i*K+2,0] == 1):
            plt.scatter(Z_data[i,0],Z_data[i,1],c='blue')        
plt.xlabel("1st significant component")
plt.ylabel("2nd significant component")
plt.title("Epoch: 0")
plt.show()
'''

start = time.time()
# Assignment: Integer Programming
# Define the solver IPOPT
opt_ipopt = pyo.SolverFactory('ipopt')


for n_iter in range(1,N_iter+1,1):
    # Define the model
    model = pyo.ConcreteModel(name='Bananced K-Means')
    #Define the variables
    model.x = pyo.Var(range(N*K), domain=pyo.Binary) 
    # Define the constraints
    model.Constraint1 = pyo.ConstraintList()
    for i in range(N):
        model.Constraint1.add(expr = model.x[i*K]+model.x[i*K+1]+model.x[i*K+2] == 1)
    # Define the objective function
    def _obj(m):
        D = dis(C[:,:,n_iter-1])
        temp2 = 0 
        for j in range (K):
            count2 = 0
            for i in range (N):   
                temp2 += m.x[i*K+j]*D[i,j]
                count2 += m.x[i*K+j]
            temp2 += l*count2*count2
        return temp2
    
    model.obj = pyo.Objective(rule = _obj, sense=pyo.minimize)
    #model.pprint()
    
    # Here we solve the optimization problem, the option tee=True prints the solver output
    result_obj_ipopt = opt_ipopt.solve(model, tee=False)
    #model.obj.display()# If using this on Google collab, we need to install the packages
    #model.x.display()
    
    obj_value[:,n_iter] = pyo.value(model.obj)
    for i in range(N*K):
        T[i,n_iter] = round(pyo.value(model.x[i]))
    n_k[:,n_iter] = count(T[:,n_iter])
    
    # Update cluster center
    for j in range(K):
        sum_Tih_xi = np.zeros((M))
        sum_Tih = 0
        for i in range(N):
            sum_Tih_xi += T[i*K+j,n_iter]*data[i]
            sum_Tih += T[i*K+j,n_iter]
        if sum_Tih != 0:
            C[j,:,n_iter] = sum_Tih_xi/sum_Tih
        else:
            C[j,:,n_iter] = C[j,:,n_iter-1]
    
    '''
    # Plot cluster result: Iris only
    for i in range(N):
            if (T[i*K+0,n_iter] == 1):
                plt.scatter(Z_data[i,0],Z_data[i,1],c='red')
            elif (T[i*K+1,n_iter] == 1):
                plt.scatter(Z_data[i,0],Z_data[i,1],c='green')
            elif (T[i*K+2,n_iter] == 1):
                plt.scatter(Z_data[i,0],Z_data[i,1],c='blue')        
    plt.xlabel("1st significant component")
    plt.ylabel("2nd significant component")
    print("Epoch: ", n_iter)
    plt.show()
    '''
    
    # If no change, break
    if (obj_value[0,n_iter] == obj_value[0,n_iter-1]):
        print("Epochs needed for convergence: ", n_iter-1)
        break

    # Balance constraint check
    if (np.linalg.norm(n_k[:,n_iter]-np.ones((K,1))*N/K)<0.01):
        print("Balance Contraint MET")
        constr_ = 1
    else:
        print("Balance Constraint UNSATISFIED")
        constr_ = 0

end = time.time()
time_ = end-start


# Plot objective function value
plt.plot(range(0,n_iter),obj_value[0,0:n_iter],c='black') 
plt.scatter(range(0,n_iter),obj_value[0,0:n_iter],c='blue')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Objective Function Value")

# Export Results to csv file
import csv

with open(r'log.csv', 'a', newline='') as csvfile:
    fieldnames = ['Obj Value','Epoch','Time','Constraint']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'Obj Value':obj_value[0,n_iter-1], 'Epoch':n_iter-1,'Time':time_,'Constraint':constr_})

'''
# Compare clustering result to the labeled data
T_true = np.zeros((N*K))
T_true[0:150:3]=1;
T_true[151:299:3]=1;
T_true[302:450:3]=1;          
Actual_C = np.zeros((K,M))
count_true = count(T_true)
for i in range (M):
    for j in range(K):
        Actual_C[j,i]=np.mean(data[(50*j):(50*j+49),i])
Min_obj_value = obj(Actual_C,T_true)
'''
