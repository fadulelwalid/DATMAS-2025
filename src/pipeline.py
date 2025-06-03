from config import setup_config, setup_models
import json
import os
from prompter import Prompter
from clean_data import Cleaner
from library import LibraryManager
class ModelPipeline:
    def __init__(self, modelname: str, main_path: str, setup: bool = False):
        setup_config(path_from_home=main_path)
        self.modelname = modelname
        self.cleaner = Cleaner()
        self.library_manager = LibraryManager(self.cleaner)

        self.models: dict = self.library_manager.load_data_json("config.json")
        self.model_configs = self.models.pop("llm_config")

        self.cur_model = self.models[modelname]
        self.cur_lib_name: str|None = None
        setup_models(selected_model=self.cur_model)

        self.prompter = Prompter(self.cur_model, self.library_manager)
        self.prompt_list = self.prompter.prompt_list


        self.libs = self.library_manager.libs #load_data_json("data_clean/libs.json")
        if setup:
            self.create_libraries()
        
    def create_libraries(self):
        self.library_manager.create_all_libraries()
        return

    def run_training(self, generate: bool = False, dataset_fname: str|None = None):
        from trainer import Trainer
        trainer = Trainer(self.models, self.modelname, self.library_manager, self.prompt_list, want_text_only=True)
        if dataset_fname: trainer.cur_dataset = dataset_fname 
        print(self.prompt_list)
        # ['manuscript_tech_docs', 'manuscript_tech_docs_example', 'generate_mcqs_tech_docs', 'training_general', 'manuscript_summary_tech_docs']
        #trainer.set_prompts(tech_prompt=self.prompt_list[1], manus_prompt=self.prompt_list[2])
        trainer.set_prompts(tech_prompt=self.prompt_list[2], manus_prompt=self.prompt_list[3])
        if generate:
            print("Generating source file before training")
            trainer.run_generate()
            print("Generated source file, now training")
        trainer.run_training()

    def run_whole_loop(self):
        # ['manuscript_tech_docs', 'manuscript_tech_docs_example', 'manuscript_tech_docs_instr_first',
        # 'generate_mcqs_tech_docs', 'training_general', 'manuscript_summary_tech_docs']
        prompt_manus = self.prompt_list[2]
        prompt_qa = self.prompt_list[3]
        if not self.prompter.is_loaded:
            self.prompter.prepare_prompter()
        self.prompter.set_prompt(prompt_manus)
        print(f"Running phase 1 for {self.cur_lib_name}")
        response_str, response = self.prompter.run_stage_one(self.cur_lib_name)
        self.prompter.set_prompt(prompt_qa)
        print(f"Running phase 2 for {self.cur_lib_name}")
        response_str = self.prompter.run_stage_two(response)
        return

    def run_all(self):

        #systems = ["system_21_tech_docs", "system_43fakkel_tech_docs"]
        systems = ["system_50_tech_docs", "system_70_tech_docs", "system_77_tech_docs"]
        self.prompter.prepare_prompter(**self.model_configs)
        for system in systems:
            self.cur_lib_name = system
            print(f" Running whole loop for system {system}")
            self.run_whole_loop()
        return
if __name__ == "__main__":
    # Have to run this file from start.py. This is due to relative imports...
    pass
