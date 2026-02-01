# Walias: A Multidimensional Workload-Aware Learned Index with Adaptive Splitting

This repository provides the implementation of **Walias**, a multidimensional workload-aware learned index structure that adaptively partitions data and queries to optimize query latency, inde cost.  
The algorithm is proposed in the paper *“Walias: A Multidimensional Workload-Aware Learned Index with Adaptive Splitting.”*

Walias introduces a **learning-driven partitioning framework** that adapts to **workload distributions** and **data characteristics**, providing significant efficiency improvements over traditional learned index methods.

---

## Features

- **Walias (LO)** and **Walias**: Two core variants for evaluating workload adaptation and learning optimization.  
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

## Datasets
`./src/datasets/dataset/`
Each workload contains **1 million data points** with corresponding workload and query sets.  
They are consistent with the four experimental distributions described in **Section 4.1** of the paper:
| Datasets | Description|
|----------|-------------|
| **TCP-H** | Based on the TPC-H benchmark, which consists of eight tables and twenty-two SQL queries. We use only the fact table lineitem, containing six attributes: shipping date, receipt date, ordered quantity, extended price before discount, discount rate, and applied tax rate for each transaction record. The dataset is scaled to 100 million tuples with a scale factor of 20. |
| **OSM** | A real-world dataset derived from the UCR STAR repository, containing 100 million POIs extracted from the northeastern region of the United States. Each record includes six attributes, such as GPS coordinates, ID, timestamp, record type, and landmark category |
| **Uniform** | A synthetic 20-dimensional dataset with 100 million objects uniformly distributed across all dimensions, where each dimension is represented by a 64-bit floating-point value. |
| **Skew** | A synthetic dataset sharing the same schema as **Uniform**,  but introducing skewness to each attribute value *v ∈ [0, 1]* using  a skew factor *r = 2*, transforming it into a larger-biased value as  *v′ = 1 − (1 − v)<sup>r</sup>* |


## Workloads

**Located in:**  
`./src/datasets/workload/`
Each type of workload contains **1000 workload for index construction, and 1000 for query test**.  
They are consistent with the four experimental distributions described in **Section 4.1** of the paper:
| Workload | Type | Description|
|----------|------|-------------|
| **UNI** | Uniform | Evenly distributed keys across dimensions |
| **GAU** | Gaussian | Clustered around a central mean with normal deviation |
| **SKE** | Skewed (Zipf-like) | Heavy-tail key distribution dominated by a few dense regions |
| **MIX** | Mixed | Hybrid of uniform and Gaussian distributions |

## Usage

To run the **Walias** and **Walias (LO)** examples and evaluate query time, construction time, and index size, follow these steps:

1. **Clone this repository**
   ```bash
   git clone https://github.com/anonymity/Walias.git
   cd Walias
   ```
2. **Compile all Java source files**
   ```bash
      javac -d bin src/**/*.java
      ```
4. **Run the main example**
   ```bash
   java -cp bin Main
      ```

This will execute the example defined in `Main.java` and output:

- Query latency  
- Index construction time  
- Index size
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
