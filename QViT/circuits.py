import torch.nn as nn
import numpy as np
import jax.numpy as jnp
import tensorcircuit as tc
import torch
from jax import pmap
from jax.numpy import array
####################################### Shared Func

# Wrapper to turn quantum circuits to functions with gradients

class QLayer(nn.Module):
    
    def circuit_to_func(self,K,quantum_circuit,nqubits):
        def f(inputs,parameters):
            return quantum_circuit(inputs,parameters,nqubits)

        f_vmap = K.vmap(f, vectorized_argnums=0)
        f_batch = tc.interfaces.torch_interface(f_vmap, jit=True)

        return f_batch

    def __init__(self,quantum_circuit,par_sizes,nqubits):
        super(QLayer,self).__init__()
        self.backend = tc.set_backend("jax")
        self.w = nn.Parameter(torch.normal(0,1/par_sizes[-1]**.5*torch.ones(par_sizes)) )
        self.f = self.circuit_to_func(self.backend,quantum_circuit,nqubits)
    def forward(self,input1):
        return self.f(input1,self.w)



########################################### Circuits in the first method
# # # # # # Circuit Architectures given in the QViT Paper. Unutilized in the project but given for the completeness

def loader_bs(X):
    qml.PauliX(wires=0)
    for i,x in enumerate(X):
        # if X[i]!=X.max():
        qml.Beamsplitter(X[i]/X.max(),0,[i,i+1])
def mmult_bs(parameters,X,length=3):

    k=0
    loader_bs(X)
    for i in range(2*length-2):
        j = length-abs(length-1-i)
        
        if i%2: 
            for _ in range(j):
                if _%2==0:
                    qml.Beamsplitter(parameters[k], 0, [_,_+1])
                    k+=1
        else:
            for _ in range(j): 
                if _%2:
                    qml.Beamsplitter(parameters[k], 0, [_,_+1])
                    k+=1
    return qml.expval(qml.PauliZ([1]))



# # # # # # # # # Circuit Architectures utilized in the project


def rbs(wires,th):
# Performs rbs operation described in the "" paper on wires with theta = th angles.
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])
    qml.CZ(wires)
    qml.RY( th,wires[0])
    qml.RY(-th,wires[1])
    qml.CZ(wires)
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])



def vector_loader(alphas,wires=None,is_x=True,is_conjugate=False):
# Loads the vector to the given wires
# alpha: the parametrized data. Parametrization can be achieved using convert_array function.
# wires: indicates which wires the operation is performed on. The default is [0,1,2,...,len(alpha)].
# is_x:  Whether to apply hadamard gate to the first gate or not. The default is True
# is_conjugate : is True if conjugate is being applied. The default is False

    if type(wires)==type(None): wires = [ i for i in range(alphas.shape[-1]+1)]
    if is_x and not(is_conjugate):qml.PauliX(wires=wires[0])
    if is_conjugate:
        for i in reversed(range(alphas.shape[-1])):
            rbs([wires[i],wires[i+1]],-alphas[...,i])
    else: 
        for i in range(alphas.shape[-1]):
            rbs([wires[i],wires[i+1]],alphas[...,i])
    if is_x and is_conjugate:qml.PauliX(wires=wires[0])


# Loads the matrix parametrized by mag_alphas and alphas to the mag_wires and wires.
def matrix_loader(mag_alphas,alphas,mag_wires,wires,is_conjugate=False):
    
# mag_alphas: The parametrized form of the rows norms of the matrix
# alphas: The parametrized form of the rows of the matrix
# mag_wires: The wires where norms of the rows are stored on.
# mag_wires: The wires where the rows are stored on.
    if not(is_conjugate):
        
        vector_loader(mag_alphas,mag_wires)
        for i in range(len(mag_wires)):
            qml.CNOT([mag_wires[i],wires[0]])
            vector_loader(alphas[i],wires,is_x=False)
            if i != len(mag_alphas):vector_loader(alphas[i+1],wires,is_x=False,is_conjugate=True)
    else:
        
        for i in reversed(range(len(mag_wires))):
            if i != len(mag_alphas):vector_loader(alphas[i+1],wires,is_x=False,is_conjugate=False)
            
            vector_loader(alphas[i],wires,is_x=False,is_conjugate=True)
            qml.CNOT([mag_wires[i],wires[0]])
            
        vector_loader(mag_alphas,mag_wires,is_conjugate=True)


def mmult(parameters,wires=None,length=None):
    
    if type(length)==type(None): length = len(wires)
    if type(wires)==type(None): wires = [ i for i in range(length)]
    k=0

    for i in range(2*length-2):
        j = length-abs(length-1-i)
        
        if i%2: 
            for _ in range(j):
                if _%2==0:
                    rbs([wires[_],wires[_+1]],parameters[k])
                    k+=1
        else:
            for _ in range(j): 
                if _%2:
                    rbs([wires[_],wires[_+1]],parameters[k])
                    k+=1


def mmult_x(parameters,wires=None,length=None):
    
    if type(length)==type(None): length = len(wires)
    if type(wires)==type(None): wires = [ i for i in range(length)]
    k=0

    for i in range(len(wires)-1):
        j = len(wires)-2-i
        
        if i==j:
            rbs([wires[j],wires[j+1]],parameters[k])
            k+=1
        else:
            rbs([wires[i],wires[i+1]],parameters[k])
            k+=1
            rbs([wires[j],wires[j+1]],parameters[k])
            k+=1
                    
                    
def compute_attention_element(inputs,parameters):
    alphas_i,alphas_j = torch.split(inputs,inputs.shape[-1]//2,dim=-1)
    wires = list(range(alphas_i.shape[-1]+2))
    qml.PauliX(wires[0])
    rbs(wires[:2],torch.pi/4)
    vector_loader(alphas_j,wires[1:],is_x=False)
    mmult(parameters,wires=wires[1:])
    vector_loader(alphas_i,wires[1:],is_conjugate=True,is_x=False)
    rbs(wires[:2],torch.pi/4)
    return qml.expval(qml.PauliZ([wires[1]]))



def compute_attention(alphas,norms,compute_element):
    yhat=[]
    n=norms.shape[1]

    n_items = alphas.shape[0]
    
    for n_i in range(n_items):
                
        res= compute_element( torch.stack([alphas[n_i,[i,j]].flatten()   for j in range(n) for i in range(n)],dim=0)  )
        e1 = (-res.reshape(n,n)/2+1/2+1e-10).sqrt()
        wij = e1*2-1
        yhat.append(wij*torch.outer(norms[n_i],norms[n_i]) )
    yhat = torch.stack(yhat,dim=0)
    return yhat




################################################################################# Circuits used in the second method

#Original encode function (basic encode):

def encode_token(circuit,data,nqubits):

    """
        Basic encoding strategy: Hadamard + Rx rotation
        
        This encoding:
        1. Applies Hadamard to create superposition
        2. Rotates around X-axis based on input data
        
        Example for 2 qubits with data [0.5, 1.0]:
        |0⟩ --H--Rx(0.5)--
        |0⟩ --H--Rx(1.0)--
    """

    for i in range(nqubits):
        circuit.H(i) # Create superposition
        circuit.rx(i,theta = data[i]) # Rotate based on data

#OUR NEW ENCODING FUNCTIONS:

def amplitude_encode(circuit, data, nqubits):
        """
        Amplitude encoding strategy: Maps data to quantum amplitudes
        
        This encoding:
        1. Normalizes input data to ensure valid quantum state
        2. Uses Rx and Ry rotations to encode data in amplitudes
        
        Example for 2 qubits with data [0.8, 0.6]:
        Normalized = [0.8/√1.0, 0.6/√1.0]
        |0⟩ --H--Rx(arcsin(0.8))--Ry(arccos(0.8))--
        |0⟩ --H--Rx(arcsin(0.6))--Ry(arccos(0.6))--
        """
        # Normalize data to ensure valid quantum state
        normalized_data = data / jnp.linalg.norm(data) 
        
        for i in range(nqubits):
            circuit.H(i)  # Create superposition
            # Encode amplitude in both X and Y rotations
            circuit.rx(i, theta=jnp.arcsin(normalized_data[i]))
            circuit.ry(i, theta=jnp.arccos(normalized_data[i]))

def phase_encode(circuit, data, nqubits):
        """
        Phase encoding strategy: Encodes data in quantum phases
        
        This encoding:
        1. Creates superposition with Hadamard
        2. Applies phase rotation based on data
        
        Example for 2 qubits with data [0.5, 1.0]:
        |0⟩ --H--Rz(0.5)--
        |0⟩ --H--Rz(1.0)--
        """
        for i in range(nqubits):
            circuit.H(i)  # Create superposition
            circuit.rz(i, theta=data[i])  # Encode in phase

def dense_angle_encode(circuit, data, nqubits):
        """
        Dense angle encoding: Uses all three rotation angles
        
        This encoding:
        1. Applies all three rotation gates (Rx, Ry, Rz)
        2. Enables encoding 3 data points per qubit
        
        Example for 1 qubit with data [0.5, 1.0, 0.7]:
        |0⟩ --Rx(0.5)--Ry(1.0)--Rz(0.7)--
        """
        for i in range(nqubits):
            idx = i * 3
            if idx < len(data):
                circuit.rx(i, theta=data[idx])
            if idx + 1 < len(data):
                circuit.ry(i, theta=data[idx + 1])
            if idx + 2 < len(data):
                circuit.rz(i, theta=data[idx + 2])
#____________________________________________________
        
# def qkv_ansatz(c,data,parameters,nqubits):

#     for i in range(nqubits):
#         c.rx(i,theta=parameters[0,i])
#     for i in range(nqubits):
#         c.ry(i,theta=parameters[1,i])
#     for i in range(nqubits-1):
#         c.cnot(i,i+1)
        
        
        
# def qkv_ansatz(c,data,parameters,nqubits):
#     for _ in range(0,parameters.shape[0]//3,3)
#     for i in range(nqubits):
#         c.rx(i,theta=parameters[_,i])
#     for i in range(nqubits):
#         c.ry(i,theta=parameters[_+1,i])
        
#     for i in range(nqubits-1):
#         c.cnot(i,i+1)
#     c.cnot(0,nqubits-1)

def qk_ansatz(circuit,data,parameters,nqubits):
    for i in range(nqubits):
        circuit.rx(i,theta=parameters[i])
    for i in range(nqubits):
        circuit.ry(i,theta=parameters[nqubits+i])

    for _ in range(2,parameters.shape[0]//nqubits+1):


        for i in range(nqubits-1):
            circuit.cnot(i,i+1)
        circuit.cnot(nqubits-1,0)
        if _ != (parameters.shape[0]//nqubits):
            for i in range(nqubits):
                circuit.ry(i,theta=parameters[nqubits*(_)+i])
        else:
            
            circuit.ry(0,theta=parameters[nqubits*(_)])
                
def v_ansatz(circuits,data,parameters,nqubits):
    for i in range(nqubits):
        circuits.rx(i,theta=parameters[i])
    for i in range(nqubits):
        circuits.ry(i,theta=parameters[nqubits+i])

    for _ in range(2,parameters.shape[0]//nqubits):


        for i in range(nqubits-1):
            circuits.cnot(i,i+1)
        circuits.cnot(nqubits-1,0)

        for i in range(nqubits):
            circuits.ry(i,theta=parameters[nqubits*(_)+i])
        
def measure_query_key(data,parameters,nqubits):
    circuit=tc.Circuit(nqubits)
    #encode_token(circuit,data,nqubits) #The following lines can be changed to a different encode function
    #amplitude_encode(circuit,data,nqubits)
    phase_encode(circuit, data, nqubits)
    #dense_angle_encode(circuit, data, nqubits)

    qk_ansatz(circuit,data,parameters,nqubits)
    return (circuit.expectation_ps(z=[0]) ).real

def measure_value(data,parameters,nqubits):
    circuit=tc.Circuit(nqubits)
    #encode_token(circuit,data,nqubits) #This following lines can be changed to a different encode function 
    #amplitude_encode(circuit,data,nqubits)
    phase_encode(circuit, data, nqubits)
    #dense_angle_encode(circuit, data, nqubits)

    v_ansatz(circuit,data,parameters,nqubits)
    return array([circuit.expectation_ps(z=[i]).real for i in range(nqubits)])
