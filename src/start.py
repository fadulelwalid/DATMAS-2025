#from prompter.pipeline import ModelPipeline
from llmware.library import Library
from library import LibraryManager
from clean_data import Cleaner
from pipeline import ModelPipeline

def main():
    model = ModelPipeline(modelname="llama", main_path=path)
    model.run_all()

def setup():
    # This function sets up the model pipeline
    model = ModelPipeline(modelname="llama", main_path=path)
    model.library_manager.delete_libraries()
    model = ModelPipeline(modelname="llama", main_path=path, setup=True)

def train_all():
    path = "master"
    #models = ["llama", "qwen", "qwen1m", "glm"]
    models = ["qwen1m"]
    for modelname in models:
        model = ModelPipeline(modelname=modelname, main_path=path)
        model.run_training(generate=True)
        del model

def run_all():
    path = "master"
    models = ["llama", "qwen", "qwen1m"]
    models = ["llama_train", "qwen_train"]
    models = ["llama", "llama_train", "qwen", "qwen_train", "qwen1m"]
    for modelname in models:
        model = ModelPipeline(modelname=modelname, main_path=path)
        model.run_all()
        del model


if __name__ == "__main__":

    #train_all()
    run_all()


    path = "master"
    #model = ModelPipeline(modelname="glm", main_path=path)
    #model.run_all()

    
    #lib_manager = LibraryManager(Cleaner())

    #model.run_training()
    #model.run_training(generate=True)
    #model.run_training(generate=True, dataset_fname="example_sys50_70_77")
    #model.run_whole_loop()
    #model.delete_libraries()

    # Changes done to llmware library
    # models.py - 11292, 7758, 11294, added "**kwargs"
    # models.py - 7682, added "embedding_dims=None,"

    # Need to do a change in venv-3.12/lib/python3.12/site-packages/llmware/models.py, line 8021. Build the core prompt with wrapper manually
    # Then change this line to cirumvent wrapping again. Fault with that it does not add the system message to the prompt

    print("Starting")