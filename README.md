# LightDTA
Help reproduce LightDTA with short guidance.
## Overview
This repository contains the implementation of LightDTA, designed for predicting drug-target binding affinities. The model focuses on efficiency while maintaining competitive performance.
## Cite
We would be honored if this work inspires or assists your research. If you find it valuable, please consider citing the following paper:
```
Huang, X., Bi, X., Xing, N. et al.
LightDTA: lightweight drug-target affinity prediction via random-walk network embedding and knowledge distillation.
Mol Divers (2026). https://doi.org/10.1007/s11030-025-11451-9
```
## Abstracts
Accurately predicting drug targeting affinity is crucial in the field of drug discovery. 
With the rapid development of artificial intelligence, many deep learning methods have been proposed for drug target affinity prediction tasks. 
However, most existing methods rely heavily on a detailed description of biochemical attributes of inputs; besides, the model architecture is getting increasingly complex just to achieve a slight performance gain. 
Together, these poses great challenges for real-world employments and applications. 

This study proposes a new lightweight framework, LightDTA, which **combines knowledge distillation and random walk algorithms** to predict drug target affinity. 
It adopts a lightweight network-based protein representation and eliminates the tedious process of collecting detailed biochemical properties. 
A knowledge distillation framework is further introduced to enrich molecular-level knowledge and enhance predictive capability while not affecting the model efficiency. 
Comprehensive experiments show that LightDTA achieves state-of-the-art performance in both classification and regression tasks, with **only 61% of the memory requirements** of the suboptimal baseline model. 
It also achieves a **7× speedup** in inference time. 
Therefore, the proposed method offers a highly efficient and accurate model for real-world prediction of drug-target affinities.

![Alt Text](https://github.com/Huang-zilin/LightDTA-final/blob/master/Figure%201.jpg)
## Requirements
Download the GitHub repo of this project onto your local server: 
```
git clone https://github.com/Huang-zilin/LightDTA-final
```
After create and activate virtual env:
```
conda create -n LightDTA python=3.7
conda activate LightDTA
```
Specified version of pytorch required:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
And install other python packages as:
```
pip install -r requirements.txt
```
💡 Note that the virtual environment and Python packages may need adjustments based on your local setup.
## Usages
#### · Project structure
```
LightDTA-final/
├── models/
│   └── [Model.py]                            - LightDTA model (for DTI task).
│   └── [Model_for_CPI.py]                    - LightDTA model (for CPI task).
│   └── [Model_teacher.py]                    - Teacher model (only used for Pre-trained).
├── results/                        
│   └── [Davis]                               - Experimental results and outputs in Davis dataset.
│   └── [KIBA]                                - Experimental results and outputs in KIBA dataset.
│   └── [Human]                               - Experimental results and outputs in Human dataset.
├── data/
│   └── [None]                                - Due to the size of data files, they are provided via external link.
├── Figure 1.jpg                              - The overall framework of LightDTA.
├── case.py                                   - Case study script for demonstration
├── create_data.py                            - Data preprocessing and feature generation.
├── requirements.txt                          - Python package dependencies.
├── test_for_CPI.py                           - A python script used to test the model on CPI tasks.
├── test_for_DTA.py                           - A python script used to test the model on DTI tasks.
├── training_for_CPI.py                       - A python script used to train the model on CPI tasks.
├── training_for_DTA.py                       - A python script used to train the model on DTI tasks.
├── training_for_teacher.py                   - Teacher model training script (only use it for Pre-trained situation).
└── utils.py                                  - Utility functions and other modules.
```
#### · Data prepared
The datasets used in this study are publicly available. 
The Davis and KIBA datasets can be obtained from their respective original sources:
```
https://github.com/hkmztrk/DeepDTA/tree/master/data
```
The Human dataset is available from the cited publications:
```
https://github.com/masashitsubaki/CPI_prediction
```
The Covid dataset is available from the cited publications:
```
https://github.com/gxCaesar/GINCM-DTA/tree/main/data
```
The Covid dataset can be obtained from its original source:
```
https://github.com/simonfqy/PADME
```
Due to the size of processed data, they can be access ```LightDTA_data``` via the following link:
```
https://mega.nz/file/YcFwnCLZ#PnNaH_YQ382JzVsIZNmyK4J6nhm1dGRmFiIz_vQoWsM
```
The processed data include:
```
data/
├── Davis / KIBA                          - DTA dataset directory for LightDTA
│   └── ligands_can.txt                   - Drug ligands information (SMILES strings).[Processed]
│   └── proteins.txt                      - Protein sequences information. [Original]
│   └── Y                                 - Binding affinity scores matrix. [Original]
│   ├── (davis/kiba)_dict.txt             - Uniprot ID mapping for each protein. [Original]
│   ├── contact_map
│   │   └── [Uniprot ID].npy              - Precomputed contact maps for proteins. [Original]
│   ├── PPI
│   │   └── ppi_data.pkl                  - PPI network data (adjacency matrix, features, protein indices). [Processed]
│   │   └── ppi_data_wv.pkl               - PPI network data after randomwalk method. [Processed]
│   │   └── ppi_data_wv.emb               - PPI network data after randomwalk method. [Processed]
│   ├── train.csv                         - Training set with drug-target-affinity pairs. [Processed]
│   ├── test.csv                          - Test set with drug-target-affinity pairs. [Processed]
│   ├── mol_data.pkl                      - Drug graph data for LightDTA input. [Processed]
│   └── pro_data.pkl                      - Protein graph data for LightDTA input. [Processed]
└── Human                                 - CPI dataset directory for LightDTA
    ├── Human.txt                          - Drug-protein interaction pairs [Original]
    ├── Human_dict.txt                      - Uniprot ID mapping for Human dataset [Processed]
    ├── contact_map
    │   └── [Uniprot ID].npy               - Protein contact maps [Original]
    ├── PPI
    │   └── ppi_data.pkl                    - PPI network data for Human proteins [Processed]
    │   └── ppi_data_wv.pkl                 - PPI network data after randomwalk method. [Processed]
    │   └── ppi_data_wv.emb                 - PPI network data after randomwalk method. [Processed]
    ├── train1.csv - train5.csv             - 5-fold cross-validation training sets. [Processed]
    ├── test1.csv - test5.csv               - 5-fold cross-validation test sets. [Processed]
    ├── mol_data.pkl                        - Drug graph data for CPI task. [Processed]
    └── pro_data.pkl                        - Protein graph data for CPI task. [Processed]
```
#### · Training workflow
After processing the data, you can retrain the model from scratch with the following command:
```
python training_for_DTA.py
python training_for_CPI.py
```
Here is the detailed introduction of the optional parameters when running:
```
--model: The model name, specifying the name of the model to be used.
 --epochs: The number of epochs, specifying the number of iterations for training the model on the entire dataset.
 --batch: The batch size, specifying the number of samples in each training batch.
 --LR: The learning rate, controlling the rate at which model parameters are updated.
 --device: The device, specifying the GPU device number used for training.
 --dataset: The dataset name, specifying the dataset used for model training.
 --num_workers: This parameter is an optional value in the Dataloader, and when its value is greater than 0, it enables 
  multiprocessing for data processing.
```
And the parameters was defaulted as:
```
--model: RWDTA
--epochs: 2000
--batch: 512
--LR: 0.0005
--device: 0
--dataset: Davis
--num_workers: 0
```
💡 Note that changes to hyperparameters may affect model performance. Please adjust them according to your local setup.
#### · Pretrained models
If you don't want to re-train the model, we provide pre-trained model as:
```
LightDTA-final/
├── results/                        
│   └── [Davis/KIBA]
│          └── [train_RWNet_best.model]
│   └── [Human]
│          └── [RWNet_Human.model]
```
Based on these pre-trained models, you can perform DTA predictions by simply running the following command:
```
python testing_for_DTA.py
python testing_for_CPI.py
```
💡 Note that before making predictions, in addition to placing the pre-trained model parameter files in the correct location, 
it is also necessary to place the required data files mentioned in the previous section in the appropriate location.
## Results
As described in paper.
## Baseline models
As described in paper.
## Contact
We welcome you to contact us by email (huangxiaoyu@stu.ouc.edu.cn) for any questions and cooperations.

