from llmware.library import Library
from llmware.configs import LLMWareConfig, PostgresConfig
from llmware.exceptions import LibraryNotFoundException

import os
import json

from clean_data import Cleaner

class LibraryManager:

    def __init__(self, cleaner):

        self.cleaner: Cleaner = cleaner

        self.systems = {k: v for k, v in self.cleaner.systems.items() if v.valid}

        self.lib_names = []
        self.phase1_lib: str|None = None # tech docs
        self.phase2_lib: str|None = None # manuscripts
        self.system_name = None
        self.system_dict = None 

        #self.load_data_json("data_clean/libs.json")
        self.libs = {} # Ordered dict of system names to data types and library names, used by trainer
        self.libraries = [x["library_name"] for x in Library().get_all_library_cards()] # Unordered list of library names, used by prompter

        self.generate_libs_name_dict()
        self.create_all_libraries()
        pass

    # LLMWare part, creates llmware libraries from provided data
    def create_library(self, lib_name, paths):
        #if lib_name in self.libraries: return
        lib = Library().create_new_library(lib_name)
        print(f"Adding files from {paths}")
        lib.add_files(paths)        
        return

    def create_libraries_from_system_dict(self, system_name):
        for data_type in self.cleaner.data_types:
            data_type_obj = getattr(self.systems[system_name], data_type)
            if data_type_obj.name in self.libraries:
                continue  # Skip if library already exists
            path = data_type_obj.clean_path
            lib_name = data_type_obj.name
            print(f"Creating library for {lib_name} with path {path}")
            self.create_library(lib_name, path)
        return

    def create_all_libraries(self):
        for system_name, _ in self.systems.items():
            self.create_libraries_from_system_dict(system_name)
        return

    def load_lib(self, lib_name):
        return Library().load_library(lib_name)


    def delete_libraries(self):
        for lib_name in self.lib_names:
            try:
                self.delete_library(lib_name)
            except LibraryNotFoundException as e:
                print(f"Library {lib_name} not found, skipping deletion.")
        return
    
    def delete_library(self, lib_name):
        lib = Library().load_library(lib_name)
        print(f"Status of deletion: {lib.delete_library(confirm_delete=True)}")
        return

    # Data part, finds all the paths to different data for each system
    def list_systems(self, data_path):
        """Returns system names to all systems in the data folder"""
        return [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]

    def get_subdirs_system(self, system_path):
        """
        Outputs a dictionary with paths to subdirectories of a system.
        Intended to be used by create_library() to add files to independent libraries, for different types of files.
        
        Example format:
        Sytem
            - Tech_docs(file)
            - Manuscript
                - Manuscript_docs(file)
            - Question
                - Question_docs(file)
        
        """
        all_dirs = {"tech_docs": [], "manuscript": [], "question": []}
        files = [x for x in os.listdir(system_path) if os.path.isfile(os.path.join(system_path, x))]
        subdirs = [x for x in os.listdir(system_path) if os.path.isdir(os.path.join(system_path, x))]

        all_dirs["tech_docs"].append(system_path)
        for x in subdirs:
            if x == "Manus": all_dirs["manuscript"].append(os.path.join(system_path, x))
            if x == "Spm": all_dirs["question"].append(os.path.join(system_path, x))
            if x == "images": all_dirs["tech_docs"].append(os.path.join(system_path, x)) #Images are tech docs

        return all_dirs

    def format_system_str(self, system: str):
        return system.replace(" ", "_").lower()

    def get_all_subdirs(self, data_path):
        systems_dict = {}
        for system in self.list_systems(data_path):
            system_path = os.path.join(data_path, system)
            system_dict: dict = self.get_subdirs_system(system_path)
            systems_dict[self.format_system_str(system)] = system_dict
        return systems_dict

    def generate_data_json(self, save_path = "data_clean"):
        data_path = self.data_path
        systems_dict = self.get_all_subdirs(data_path)
        with open(os.path.join(save_path, "data.json"), "w") as f:
            json.dump(systems_dict, f)
        return

    def generate_libs_name_dict(self, save_path: str = "data_clean"):

        for system_name, system_dict in self.systems.items():
            self.libs.update({system_dict.name: {}})
            for data_type in self.cleaner.data_types:
                lib_name = f"{self.format_system_str(system_name)}_{data_type}"
                datatype_obj = getattr(system_dict, data_type)
                datatype_obj.name = lib_name
                self.lib_names.append(lib_name)
                self.libs.get(system_dict.name).update({data_type: lib_name})
        return

    def load_data_json(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        return data




if __name__ == "__main__":
    from config import setup_config
    setup_config("master")

    #generate_data_json("data", "data_clean")
    #generate_libs_name_dict(load_data_json("data_clean/data.json"), "data_clean")


    lib_manager = LibraryManager("data_clean", Cleaner())



    #data = load_data_json("data_clean/data.json")
    #new_data = {"system_77": {"tech_docs": ["data/System 77"], "manuscript": ["data/System 77/Manus"], "question": ["data/System 77/Spm"]}}
    #new_data = {"system_77": data["system_77"], "system_70": data["system_70"], "system_50": data["system_50"]}
    #create_library("system3_77_tech_docs", ["data_clean/System 77"])
    #create_library("system_77_tech_docs", ["data_clean/System 77"])
    #create_libraries_from_system_dict("system_77", new_data["system_77"])

    #{"tech_docs": "system_70_tech_docs", "manuscript": "system_70_manuscript", "question": "system_70_question"}
    #create_libraries_from_system_dict("system_77", new_data["system_77"])
    #create_libraries_from_system_dict("system_50", new_data["system_50"])

    #create_all_libraries(data)