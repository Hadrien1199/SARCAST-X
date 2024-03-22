
# SARCAST-X - Copenhagen Business School thesis project on sarcasm and fake-news detection in climatechange-related tweets (X)
## Folders:

TO DO BEFORE FIRST RUN:


TO DO AFTER CHANGING ENVIRONMENT VARIABLES:

## Environment Variables

In the `params.py` file, set the following variables:

- `SARCASM_MODEL_PATH`: Path to the sarcasm model on huggingface
- `FAKE_NEWS_MODEL_PATH`: Path to the fake news model on huggingface
- `CUDA`= False if you have a GPU and do not want to use it, True otherwise


## Model Architectures

- *model_functions.py* contains all the model finetuning functions that are imported in *main.py*

## Model Save and MLFLOW logic

- *registery.py* contains the logic to save the model in the MLFLOW registry and load existing models from the registry.

## Running the Code

To run the code, execute *python ml_logic/main.py* or *make run main*
