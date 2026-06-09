# Hugging Face 101 & Data Analysis Workshop

This repository contains the hands-on materials prepared for the **NEST × BAU AI Club Hugging Face 101 & Data Analysis Workshop**.

The workshop introduces participants to the Hugging Face ecosystem through practical examples involving dataset loading, data preprocessing, sentiment analysis, facial expression analysis, Gradio application development, and Hugging Face Spaces deployment.

The materials are organized as a step-by-step learning path. Participants first explore how preprocessing decisions affect machine learning results, then use pretrained Hugging Face models for sentiment analysis and facial expression analysis. Finally, they combine text-based sentiment analysis and image-based facial expression analysis in a Gradio application that can be deployed to Hugging Face Spaces.

## Repository Structure

```text
workshop/
│
├── README.md
│
├── hf_spaces/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
│
├── notebooks_finished/
│   ├── photos/
│   ├── 01_data_preprocessing_ml.ipynb
│   ├── 02_sentiment_analysis_hf.ipynb
│   ├── 03_facial_expression_analysis_hf.ipynb
│   └── 04_sentiment_facial_expression_gradio.ipynb
│
└── notebooks_unfinished/
    ├── 01_data_preprocessing_ml.ipynb
    ├── 02_sentiment_analysis_hf.ipynb
    ├── 03_facial_expression_analysis_hf.ipynb
    └── 04_sentiment_facial_expression_gradio.ipynb
```

## Folder Descriptions

### `notebooks_unfinished/`

This folder contains the starter versions of the workshop notebooks.

Participants can use these notebooks during the live session to follow the implementation step by step. Some cells may be incomplete or intentionally left for hands-on practice.

### `notebooks_finished/`

This folder contains the completed versions of the workshop notebooks.

These notebooks can be used after the workshop for review, self-study, or comparison with the participants' own implementations.

### `notebooks_finished/photos/`

This folder is reserved for sample images used in the facial expression analysis notebook.

Participants may use their own images only with appropriate consent. The examples in this workshop are intended for educational demonstration purposes.

### `hf_spaces/`

This folder contains the files required to deploy the final Gradio application to Hugging Face Spaces.

It includes:

* `app.py`: the main Gradio application
* `requirements.txt`: required Python packages for the Space
* `README.md`: Hugging Face Space description and metadata

## Workshop Notebooks

### 1. Data Preprocessing and Machine Learning

Notebook:

```text
01_data_preprocessing_ml.ipynb
```

This notebook demonstrates how preprocessing choices affect machine learning results.

Topics include:

* Loading a dataset
* Exploring data structure
* Detecting missing values
* Removing rows with missing values
* Filling missing values using imputation
* Scaling numerical features
* Applying dimensionality reduction
* Training a simple machine learning model
* Comparing model results across preprocessing strategies

The goal is not to build the best possible predictive model, but to understand how preprocessing decisions influence data size, information loss, and model performance.

### 2. Sentiment Analysis with Hugging Face

Notebook:

```text
02_sentiment_analysis_hf.ipynb
```

This notebook introduces sentiment analysis using Hugging Face datasets and pretrained models.

Topics include:

* Loading a text dataset from Hugging Face
* Inspecting review or comment data
* Using a pretrained sentiment analysis model
* Running inference on sample texts
* Interpreting model labels and confidence scores

### 3. Facial Expression Analysis with Hugging Face

Notebook:

```text
03_facial_expression_analysis_hf.ipynb
```

This notebook demonstrates image-based facial expression analysis using a pretrained Hugging Face model.

Topics include:

* Loading an image
* Using an image classification pipeline
* Predicting facial expression categories
* Interpreting model outputs
* Discussing the limitations of facial expression analysis

The model output should be interpreted as a classification of visible facial expression patterns, not as a reliable measurement of a person's true emotional state.

### 4. Sentiment + Facial Expression Gradio Demo

Notebook:

```text
04_sentiment_facial_expression_gradio.ipynb
```

This notebook combines text-based sentiment analysis and image-based facial expression analysis in a simple Gradio interface.

Topics include:

* Creating a Gradio interface
* Accepting text and image inputs
* Running sentiment analysis on text
* Running facial expression analysis on images
* Combining outputs using simple rule-based logic
* Preparing the application for Hugging Face Spaces deployment

## Hugging Face Spaces Application

The `hf_spaces/` folder contains the final deployable version of the Gradio application.

To run it locally:

```bash
cd hf_spaces
pip install -r requirements.txt
python app.py
```

To deploy it to Hugging Face Spaces:

1. Create a new Space on Hugging Face.
2. Select **Gradio** as the SDK.
3. Upload the following files from the `hf_spaces/` folder:

   * `app.py`
   * `requirements.txt`
   * `README.md`
4. Wait for the Space to build and run.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Install the required packages:

```bash
pip install -r hf_spaces/requirements.txt
```

For notebook execution, additional common packages may be required depending on the environment:

```bash
pip install datasets transformers scikit-learn pandas numpy matplotlib pillow gradio torch
```

Alternatively, the notebooks can be opened and executed in Google Colab.

## Learning Outcomes

By the end of this workshop, participants will be able to:

1. Navigate the Hugging Face ecosystem.
2. Load datasets using the `datasets` library.
3. Perform basic data preprocessing and cleaning.
4. Compare deletion-based and imputation-based missing value handling.
5. Apply scaling and dimensionality reduction.
6. Use pretrained Hugging Face models for sentiment analysis.
7. Use pretrained image classification models for facial expression analysis.
8. Build simple interactive applications with Gradio.
9. Prepare a Gradio application for Hugging Face Spaces deployment.
10. Discuss the ethical limitations of emotion-related AI systems.

## Ethical Notice

This workshop is intended for educational purposes only.

Facial expression analysis does not determine a person's true emotional state. It only classifies visible facial expression patterns based on the model's training data and may produce biased, incorrect, or context-insensitive outputs.

The applications in this repository should not be used for psychological, medical, legal, educational assessment, recruitment, security, or other high-stakes decision-making purposes.

Participants should only use images with appropriate consent. Images used during the workshop should not be stored, shared, or reused without permission.

## Suggested Citation

If you use or adapt these materials, please cite the workshop repository as:

```text
Çakır, D. Hugging Face 101 & Data Analysis Workshop. NEST × BAU AI Club Workshop Materials.
```

## License

This repository is shared for educational purposes.

Please check the licenses and terms of use of all datasets, pretrained models, and third-party libraries used in the notebooks and applications before reuse in other projects.
