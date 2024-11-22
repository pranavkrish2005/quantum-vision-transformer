# Hybrid Quantum Vision Transformer

This project expands upon the research done in the paper Hybrid Quantum Vision Transformers for Event Classification in High Energy Physics (https://arxiv.org/abs/2402.00776).


## Quantum encoding
The specific changes are made to the quantum circuit located in the circuits.py file. These include 3 additional encoding functions. These additional encoding methids and their benefits are explained in the paper Quantum Data Encoding: A Comparative Analysis of
Classical-to-Quantum Mapping Techniques and Their Impact on Machine Learning Accuracy (https://arxiv.org/pdf/2311.10375). An outline of these functions are explained below:

The original encoding function (encode_token) consists of a Hadamard + Rx rotation on the input data and is 1 data point per qubit.
This encoding:
  1. Applies Hadamard to create superposition
  2. Rotates around X-axis based on input data
        
  Example for 2 qubits with data [0.5, 1.0]:
  |0⟩ --H--Rx(0.5)--
  |0⟩ --H--Rx(1.0)--

The first new encoding function is amplitude encoding (amplitude_encode), which maps data to quantum amplitudes. It consists of 2 data points per qubit. It is good for better preservation of data relationships and more precise control over the quantum state.
This encoding:
  1. Normalizes input data to ensure valid quantum state
  2. Uses Rx and Ry rotations to encode data in amplitudes
        
  Example for 2 qubits with data [0.8, 0.6]:
  Normalized = [0.8/√1.0, 0.6/√1.0]
  |0⟩ --H--Rx(arcsin(0.8))--Ry(arccos(0.8))--
  |0⟩ --H--Rx(arcsin(0.6))--Ry(arccos(0.6))--

The second new ecnoding function is phase encoding (phase_encode), which encodes data in quantum phases. It consists of 1 data point per qubit. It is good for pattern recognition tasks and interference-based algorithms. Note that this is the encoding function being run in the current code.
This encoding:
  1. Creates superposition with Hadamard
  2. Applies phase rotation based on data
        
  Example for 2 qubits with data [0.5, 1.0]:
  |0⟩ --H--Rz(0.5)--
  |0⟩ --H--Rz(1.0)--

The third new encoding function is dense angle encoding (dense_angle_encode), whcih uses all three rotation angles. It consists of 3 data points per qubit. It is good for maximum data density, more efficient use of resources, and better for complex data patterns. 
This encoding:
  1. Applies all three rotation gates (Rx, Ry, Rz)
  2. Enables encoding 3 data points per qubit
        
  Example for 1 qubit with data [0.5, 1.0, 0.7]:
  |0⟩ --Rx(0.5)--Ry(1.0)--Rz(0.7)--


## Hyperparameter tuning
  1. increasing the complexity of the free forward Neural Network.
  Adding more layers and adding more neurons to each layer in the Free Forward Neural Network part of the transformer model, increased the model's accuray and decreased the model loss significantly.

NOTE: We also found that adding more neurons does not make the model better since it becomes prone to overfitting and does not perform well enough for testing data.

  2. Increasing the number of q_bits to caluclate the Key, Value and Query in the attention layer.
  THe code originally had only 2 qbits and we changed that making it similar to the paper cited above. Just making this simple change resulted in each attention head being able to encode the positioning and meaning of each of the pixel more accurately, increasing the overall performance of the model.

## Circuit Changes