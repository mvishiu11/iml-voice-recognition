# IML Voice Recognition Project

This project aims to develop a machine learning module for voice-based recognition using Convolutional Neural Networks (CNNs) trained on spectrograms generated from voice recordings.

## Setup Instructions

### Requirements
- Python 3.12 or above
- See `requirements.txt` or use the provided Conda environment in `environment.yml`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iml-voice-recognition.git
   cd iml-voice-recognition
```
4. Create and activate the environment:
5. ```bash
   conda env create -f environment.yml
   conda activate iml-voice-recognition
   ```
6. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```
7. Run Jupyter notebooks:

   ```bash
   jupyter lab
   ```

### Usage

- The `notebooks/` folder contains exploratory and model training notebooks.
- Use `tensorboard` to track experiments:
  ```bash
  tensorboard --logdir=runs
- You can see the previous experiments on tensorboard by running the following command:
  ```bash
  tensorboard --logdir=logs
  ```

### Milestone 1

Results of the first milestone can be found in the `notebooks/` folder. Steps completed in the first milestone include:
- Data collection and preprocessing
- Data visualization (of the spectrograms)
- Defining the CNN model architecture
- Training the model on the dataset
- Evaluating the model on the test set
Detailed results of the training can be found on TensorBoard by running the following command:
```bash
tensorboard --logdir=logs
```

### Contributing

1. Run pre-commit hooks before committing changes:
   ```bash
   pre-commit run --all-files
   ```

2. All commits should have meanigful names and semantic prefixes:
   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation changes
   - `style`: Code style changes
   - `refactor`: Code refactor
   - `test`: Add or modify tests
   - `chore`: Maintenance tasks

3. Commit to dev branch and create a pull request for review.
