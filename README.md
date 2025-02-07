# Project 1: End-to-End Pipeline to Classify News Articles

## Table of Contents
1. [How to Run the Code](#how-to-run-the-code)
2. [Dependencies](#dependencies)
3. [File Structure](#file-structure)
4. [Authors](#authors)

---

## How to Run the Code
Necessary steps to run the code:

1. Navigate to the project directory:
   ```bash
   cd Project1
   ```
   
2. Place the raw dataset file and GLoVE files in the `data/` folder as mentioned in [File Structure](#file-structure).

3. Install the required dependencies:
   ```bash
   conda env create -f environment.yml
   ```

4. For each question, simply run the script with corresponding name. For example, to get the answer of question 1, run the following script:
   ```bash
   python code/proj_q1.py
   ```

5. To ensure the code runs smoothly, please execute python scripts in the order of the questions.

---

## Dependencies
This project requires the following libraries and tools:

- Python <3.13
- NumPy
- pandas
- matplotlib
- scikit-learn
- umap-learn
- nltk

---

## File Structure
This Project is organized as follows:
```bash
├── data/                 # Original dataset files
│   ├── Project1-ClassifucationDataset.csv         
│   ├── glove.6B.50d.txt  
│   ├── glove.6B.100d.txt  
│   ├── glove.6B.200d.txt  
│   └── glove.6B.300d.txt
├── processed_data/       # Processed data  
├── code/                 # Source code
│   ├── proj1_q1.py         
│   ├── proj1_q2_q3.py
│   ├── proj1_q4_q5.py
│   ├── proj1_q6.py
│   ├── proj1_q7.py
│   ├── proj1_q8.py
│   ├── proj1_q9_1.py
│   ├── proj1_q9_2.py
│   ├── proj1_q9_3.py
│   ├── proj1_q11.py
│   ├── proj1_q12.py
│   └── proj1_q13.py       
├── README.md            # Documentation
└── environment.yml      # Python dependencies file
```

Notes:
- Place the .csv file and GLoVE files in the `data/` folder and **DO NOT** change their names.
- All source code is located in the `code/` folder, with functionality coresponding to its name.
- Some processed data will be saved in the `processed_data/` folder.

---

## Authors

This project was collaboratively developed by the following contributors:

| Name                | UID                       |  Contact             |
|---------------------|---------------------------|----------------------|
| **LiWei Tan**       | Project Manager           | alice@example.com    |
| **TianXiang Xing**  | Project Manager           | alice@example.com    |
| **Junrui Zhu**      | 606530444                 | zhujr24@g.ucla.edu   |

---