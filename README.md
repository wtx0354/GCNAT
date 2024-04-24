# GCNATMDA: Microbe-Drug Association Prediction

GCNATMDA is a computational model designed to predict potential associations between microbes and drugs. It leverages the power of graph convolutional networks and graph attention networks to analyze complex interaction networks, based on datasets such as MDAD, Drugvirus, and aBiofilm. These datasets contain associative and similarity files relevant to microbe-drug interactions.

## Installation

To run GCNATMDA, you need the following software:

- **Tensorflow 1.15**
- **Python 3.6**

Ensure these are properly installed on your system before proceeding with the execution of the program.

## Usage

After setting up the required paths correctly, you can start the program by running the `main` script. This can typically be done from the command line:

```bash
python main.py
```

## Data Files

The datasets used in this project include:
- **MDAD**
- **Drugvirus**
- **aBiofilm**

Each dataset comprises files that detail both the associations and similarities necessary for the microbe-drug association prediction.

## Contributing

Contributions to GCNATMDA are welcome. If you have suggestions for improvement or want to report bugs, please open an issue or submit a pull request.

