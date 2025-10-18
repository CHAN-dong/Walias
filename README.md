# Walas-Tree: A Multidimensional Workload-Aware Learned Tree with Adaptive Splitting

This repository provides the implementation of **Walas-Tree**, a multidimensional workload-aware learned index structure that adaptively partitions data and queries to optimize query latency, index size, and construction time.  
The algorithm is proposed in the paper *“Walas-Tree: A Multidimensional Workload-Aware Learned Tree with Adaptive Splitting.”*

Walas-Tree introduces a **learning-driven partitioning framework** that adapts to **workload distributions** and **data characteristics**, providing significant efficiency improvements over traditional learned index methods.

---

## Features

- **Walas-Tree (LO)** and **Walas-Tree**: Two core variants for evaluating workload adaptation and learning optimization.  
- **Adaptive Splitting Strategy**: Dynamically adjusts data partition boundaries based on workload characteristics.  
- **Workload-Aware Cost Model**: Balances query latency and index size via learned workload feedback.  
- **Visualization Support**: View data distribution, query patterns, and partitioning results interactively.  
- **Scalable Evaluation**: Includes 1M-scale datasets for performance benchmarking.  

---

## Technology Stack

- **Language**: Java 11+  
- **Random Forest**: Weka library
- **Visualization**: Java AWT / Swing-based visualization utilities  
- **Data Management**: Custom multidimensional data structure for adaptive indexing  

---

## Usage

To run the **Walas-Tree** and **Walas-Tree (LO)** examples and evaluate query time, construction time, and index size, follow these steps:

1. **Clone this repository**
   ```bash
   git clone https://github.com/anonymity/Walas-Tree.git
   cd Walas-Tree
   
2. **Compile all Java source files**
    javac -d bin src/**/*.java

3. **Run the main example**
   java -cp bin Main

## Configuration

The default runtime parameters are defined in:
./src/data_structure/Parameter.java

You can modify:
    Dataset selection
    Query workload type
    Partition parameters
    Learning rates and thresholds

To visualize dataset and workload distributions:

java -cp bin utils.DatasetVisualizer
