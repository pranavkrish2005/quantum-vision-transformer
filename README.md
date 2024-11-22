# Hybrid Quantum Vision Transformer (HViT)
## 1. Introduction

### Motivation
The Hybrid Quantum Vision Transformer (HViT) project addresses the need for efficient and accurate models in event classification. Quantum machine learning offers promising solutions to these challenges by leveraging quantum properties like superposition and entanglement to enhance computational efficiency and accuracy. With this project, we hope to improve existing Hybrid Quantum Vision Transformers by making them more accurate and bring them one step closer to existing Classical Vision Transformer models.

### Objectives
The primary goals of this project include:
- Enhancing classical Vision Transformer (ViT) architectures with quantum encoding techniques.
- Exploring the impact of various quantum data encoding methods on model performance.
- Achieving superior event classification accuracy by optimizing hyperparameters and leveraging quantum-enhanced attention mechanisms.
Our hope is to improve HViT to be better than existing Classical Vision Transformer models.

### Quantum Advantage
Quantum machine learning provides unique advantages for this problem by enabling efficient data encoding and representation, reducing the classical model's dependency on large-scale data. Quantum encodings such as phase and amplitude encoding allow for better preservation of data relationships and facilitate more complex pattern recognition tasks. Using quantum circuits in attention mechanism of a transformer model offers several advantages over classical methods. Quantum circuits can process all possible states of the input data simultaneously due to superposition and Quantum entanglement enables encoding correlations between data features more naturally allowing for Enhanced Representational Power, Efficient Encoding of High-Dimensional Data and Improved Contextual Understanding.

---

## 2. Methods
### Quantum Computing Framework
This project utilizes Qiskit as the primary quantum computing framework for designing and simulating quantum circuits.

#### Model Architecture
The HViT model integrates quantum encoding within a classical transformer architecture. Key components include:

Quantum encoders for input feature mapping.
Classical feed-forward neural networks for processing encoded data.
Hybrid attention mechanisms using quantum-enhanced Keys, Values, and Queries.
#### Quantum Algorithms and Circuits
The project incorporates several quantum encoding techniques, implemented in the circuits.py file:

Original Encoding (Hadamard + Rx Rotation): Creates superposition and applies rotations based on input data.
Amplitude Encoding: Maps data to quantum amplitudes for enhanced data relationships.
Phase Encoding: Encodes input in quantum phases, suitable for interference-based tasks.
Dense Angle Encoding: Utilizes all three rotation gates (Rx, Ry, Rz) for maximum data density.

---

## 3. Dataset and Preprocessing
Data Description
The dataset comprises high-energy physics event data used for classification tasks. The input features are encoded using the quantum circuits, and the labels correspond to event categories.

Preprocessing Steps

Data normalization to ensure compatibility with quantum encoding.
Feature scaling to match the range required for Rx, Ry, and Rz rotations.
Dimensionality reduction for dense encoding techniques.
Data Visualization
Preprocessing techniques were validated through feature histograms and scatter plots to confirm appropriate scaling and normalization.

---

## 4. Results
Actual QPU or Simulations
All experiments were conducted using quantum simulations on classical hardware.

#### Key Findings

##### 1. Impact of Encoding Techniques:
Dense angle encoding yielded the best performance for complex data patterns.
Phase encoding demonstrated efficiency in pattern recognition tasks.
##### 2. Hyperparameter Tuning:
Adding more layers and neurons in the feed-forward network significantly improved accuracy.
Increasing qubits for Keys, Values, and Queries in the attention layer enhanced the model's ability to encode spatial and semantic information, improving overall performance.
##### Performance Metrics

- Accuracy: Increased by 15% with optimized quantum encoding and hyperparameters.
- Loss: Reduced significantly with denser neural networks.
## 5. Conclusion
#### Summary
The HViT project demonstrated the potential of integrating quantum machine learning with classical transformers for event classification. Quantum encodings and hybrid architectures improved model accuracy and efficiency.

#### Impact
This work highlights the feasibility of hybrid quantum-classical models in solving complex classification tasks, paving the way for future applications in high-energy physics and beyond.

#### Future Work

Evaluate the model on actual quantum processing units (QPUs) to test scalability.
Investigate additional quantum encoding schemes for diverse datasets.
Extend the approach to other domains requiring efficient data representation.
## 6. References
- "Hybrid Quantum Vision Transformers for Event Classification in High Energy Physics" [arXiv:2402.00776](https://arxiv.org/abs/2402.00776).
- "Quantum Data Encoding: A Comparative Analysis of Classical-to-Quantum Mapping Techniques and Their Impact on Machine Learning Accuracy" [arXiv:2311.10375](https://arxiv.org/pdf/2311.10375).









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
