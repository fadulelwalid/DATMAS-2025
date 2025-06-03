import os
import json
import shutil
from system import System, DataType
from manus_extractor import run_manus_extractor, data_to_json
class Cleaner:
    """
    Class is intended to move data from an ingest folder to a cleaned folder.
    Selectively moves data based on file types, selects only 1 of each type of file in each system.
    """
    def __init__(self, data_root: str = "data", cdata_root: str = "data_clean"):
        self.data_root: str = data_root
        self.cdata_root: str = cdata_root
        self.systems: dict = {}
        self.find_systems()
        self.remove_files()
        #self.remove_pdf_files()
        self.check_system_validity()
        self.create_clean_dirs()

        self.find_priority()
        self.select_one_manus()
        self.select_one_question()
        self.select_one_tech_doc()
        
        self.copy_files()

        self.invalid_systems: list = [k for k, v in self.systems.items() if v.valid == False]
        self.untrainable_systems: list = [k for k, v in self.systems.items() if v.trainable == False]
        self.valid_trainable_systems: list = [k for k, v in self.systems.items() if v.valid == True and v.trainable == True]

        self.data_types = ["tech_docs", "manuscript", "question"]

    def find_systems(self):
        """
        Find all systems in the data root directory and clean data root directory.

        Gathers all files in the system directory. 
        Is prone to error if the system directory does not contain the expected subdirectories; "Manus" and "Spm"
        Expects tech_docs to be in the system directory and not in a subdirectory
        """

        systems = {x: System(
            name = x.replace(" ", "_").lower(),
            path = os.path.join(self.data_root, x),
            clean_path = os.path.join(self.cdata_root, x)
        ) for x in os.listdir(self.data_root)}

        for _, system_dict in systems.items():
            system: System = system_dict
            if not os.path.exists(os.path.join(system.path, "Manus")):
                system.valid = False
                continue
            if not os.path.exists(os.path.join(system.path, "Spm")):
                system.valid = False
                continue
            system.tech_docs = DataType(
                type = "tech_docs",
                path = system.path,
                clean_path = system.clean_path,
                files = [x for x in os.listdir(system.path) if os.path.isfile(os.path.join(system.path, x))]
            )
            if os.path.exists(system.tech_docs.clean_path):
                system.tech_docs.clean_files = [x for x in os.listdir(system.tech_docs.clean_path) if os.path.isfile(os.path.join(system.clean_path, x))]
            system.manuscript = DataType(
                type = "manuscript",
                path = os.path.join(system.path, "Manus"),
                clean_path = os.path.join(system.clean_path, "Manus"),
                files = [x for x in os.listdir(os.path.join(system.path, "Manus")) if os.path.isfile(os.path.join(system.path, "Manus", x))]
            )
            if os.path.exists(system.manuscript.clean_path):
                system.manuscript.clean_files = [x for x in os.listdir(system.manuscript.clean_path) if os.path.isfile(os.path.join(system.manuscript.clean_path, x))]
            system.question = DataType(
                type = "question",
                path = os.path.join(system.path, "Spm"),
                clean_path = os.path.join(system.clean_path, "Spm"),
                files = [x for x in os.listdir(os.path.join(system.path, "Spm")) if os.path.isfile(os.path.join(system.path, "Spm", x))]
            )
            if os.path.exists(system.question.clean_path):
                system.question.clean_files = [x for x in os.listdir(system.question.clean_path) if os.path.isfile(os.path.join(system.question.clean_path, x))]
        self.systems = systems
        return
    
    def check_system_validity(self):
        """
        Check if the system is valid. 
        A system is valid if it contains the subdirectories "Manus" and "Spm"
        Also valid if the subdirectories contain files.
        Checks whether the system is transferred to the clean data folder.
        """

        # Loop through all systems and check if they are valid
        for _, system_dict in self.systems.items():
            system: System = system_dict
            if not system.valid:
                continue
            if not os.path.exists(os.path.join(system.path, "Manus")):
                system.valid = False
                continue
            if not os.path.exists(os.path.join(system.path, "Spm")):
                system.valid = False
                continue
            if len(system.tech_docs.files) == 0:
                system.valid = False
                continue
            if len(system.manuscript.files) == 0:
                system.valid = False
                continue
            if len(system.question.files) == 0:
                system.valid = False
                continue

        # Check if the system is already copied to the clean data folder
        for _, system_dict in self.systems.items():
            system: System = system_dict
            if not system.valid:
                continue
            if len(system.tech_docs.clean_files) == 0:
                system.should_copy = True
                continue
            if len(system.manuscript.clean_files) == 0:
                system.should_copy = True
                continue
            if len(system.question.clean_files) == 0:
                system.should_copy = True
                continue
        return
    
    def create_clean_dirs(self):
        """
        Create the clean data directories.
        Updates the system dictionary with the clean path for each system.
        Creates the clean data directories for each system.
        Creates the clean data directories for each system type.
        """
        if not os.path.exists(self.cdata_root):
            os.makedirs(self.cdata_root)
        for _, system_dict in self.systems.items():
            system: System = system_dict
            # Skip if system is defective / not valid
            if not system.valid:
                continue

            system_path = system.clean_path
            manus_path = os.path.join(system_path, "Manus")
            question_path = os.path.join(system_path, "Spm")

            system.tech_docs.clean_path = system_path
            system.manuscript.clean_path = manus_path
            system.question.clean_path = question_path

            if not os.path.exists(system_path):
                os.makedirs(system_path)
            if not os.path.exists(manus_path):
                os.makedirs(manus_path)
            if not os.path.exists(question_path):
                os.makedirs(question_path)
        return
    
    def remove_files(self):
        filetypes = [".msg", ".pdf"]
        tech_doc_files = []
        manuscript_files = []
        question_files = []
        files_to_remove = []
        # A msg file popped up, remove it
        for _, system_dict in self.systems.items():
            system: System = system_dict
            # Skip if system is defective / not valid
            if not system.valid:
                continue
            for filetype in filetypes:
                # Check if the filetype is in the tech_docs, manuscript or question files
                for tech_doc_file in system.tech_docs.files:
                    if tech_doc_file.lower().endswith(filetype):
                        files_to_remove.append(tech_doc_file)

                        
                for manuscript_file in system.manuscript.files:
                    if manuscript_file.lower().endswith(filetype):
                        files_to_remove.append(manuscript_file)


                for question_file in system.question.files:
                    if question_file.lower().endswith(filetype):
                        files_to_remove.append(question_file)

            # Update the system to exclude the files to remove
            system.tech_docs.files = [f for f in system.tech_docs.files if f not in files_to_remove]
            system.manuscript.files = [f for f in system.manuscript.files if f not in files_to_remove]
            system.question.files = [f for f in system.question.files if f not in files_to_remove]
        return
    
    # Deprecated function, not used anymore (pdf files considered in remove_files)
    def remove_pdf_files(self):
        """
        Remove all pdf files which should not be transferred to clean data folder.
        """
        for _, system_dict in self.systems.items():
            system: System = system_dict
            # Skip if system is defective / not valid
            if not system.valid:
                continue
            system.tech_docs.files = [x for x in system.tech_docs.files if not x.lower().endswith(".pdf")]
            system.manuscript.files = [x for x in system.manuscript.files if not x.lower().endswith(".pdf")]
            system.question.files = [x for x in system.question.files if not x.lower().endswith(".pdf")]
        return
    
    def find_priority(self):
        priority = [".doc", ".pptx", ".docx", ".jsonl"]
        for _, system_dict in self.systems.items():
            system: System = system_dict
            # Skip if system is defective / not valid
            if not system.valid:
                continue
            # Check if there are multiple files with the same priority extension. Adds them to the list
            elements = [system.manuscript, system.question, system.tech_docs]
            for element in elements:
                files = element.files
                        # Select the file with the highest priority
                selected_file: str = max(files, key = lambda x: priority.index(os.path.splitext(x)[1].lower()))
                selected_fileext = os.path.splitext(selected_file)[1].lower()
                element.filetype = selected_fileext
        return
    def select_one_manus(self):
        for _, system_dict in self.systems.items():
            system: System = system_dict
            # Skip if system is defective / not valid
            if not system.valid:
                continue
            if len(system.manuscript.clean_files) > 0:
                # If the system already has cleaned manuscript files, skip conversion
                continue
            files = system.manuscript.files
            selected_fileext = system.manuscript.filetype
            files_to_convert = [x for x in files if x.lower().endswith(selected_fileext)]
            # Check if there are multiple files with the same priority extension. Adds them to the list
            for file in files_to_convert:
                new_file_name = f"{os.path.splitext(file)[0]}.jsonl"
                file_path = os.path.join(system.manuscript.path, file)
                clean_file_path = os.path.join(system.manuscript.clean_path, new_file_name)
                
                data = self.convert_selected_files(file_path)
                if len(data) == 0:
                    data = self.convert_selected_files(file_path, section_idx=0, manus_idx=1)
                data_to_json(data, clean_file_path)

                system.manuscript.file_paths.append(file_path)
                system.manuscript.clean_file_paths.append(clean_file_path)
            system.manuscript.converted = True
            if not system.manuscript.converted:
                system.manuscript.file_paths = [os.path.join(system.manuscript.path, x) for x in files if os.path.splitext(x)[1].lower() == selected_fileext]
                system.manuscript.clean_file_paths = [os.path.join(system.manuscript.clean_path, x) for x in files if os.path.splitext(x)[1].lower() == selected_fileext]
            if len(system.manuscript.file_paths) > 0:
                system.should_copy = True
            if len(system.manuscript.clean_file_paths) > 1:
                system.trainable = False
                print(f"More than one manuscript file with the same extension in {system.name}")
        return

    def select_one_question(self):
        priority = [".doc", ".docx", ".pptx", ".jsonl"]
        for _, system_dict in self.systems.items():
            system: System = system_dict
            # Skip if system is defective / not valid
            if not system.valid:
                continue
            files = system.question.files
            # Select the file with the highest priority
            selected_file: str = max(files, key = lambda x: priority.index(os.path.splitext(x)[1].lower()))
            selected_fileext = os.path.splitext(selected_file)[1].lower()
            system.question.filetype = selected_fileext
            # Check if there are multiple files with the same priority extension. Adds them to the list
            system.question.file_paths = [os.path.join(system.question.path, x) for x in files if os.path.splitext(x)[1].lower() == selected_fileext]
            system.question.clean_file_paths = [os.path.join(system.question.clean_path, x) for x in files if os.path.splitext(x)[1].lower() == selected_fileext]
            if len(system.question.file_paths) > 1:
                system.trainable = False
                print(f"More than one question file with the same extension in {system.name}")
        return
    
    def select_one_tech_doc(self):
        """
        Select all valid tech docs for each system.
        Selects the files with the highest priority.
        Sets untrainable if there are multiple files with the same priority."""
        priority = [".doc", ".docx", ".jsonl"]
        for _, system_dict in self.systems.items():
            system: System = system_dict
            # Skip if system is defective / not valid
            if not system.valid:
                continue
            files = system.tech_docs.files
            # Select the file with the highest priority
            selected_file: str = max(files, key = lambda x: priority.index(os.path.splitext(x)[1].lower()))
            selected_fileext = os.path.splitext(selected_file)[1].lower()
            system.tech_docs.filetype = selected_fileext
            # Check if there are multiple files with the same priority extension. Adds them to the list
            system.tech_docs.file_paths = [os.path.join(system.tech_docs.path, x) for x in files if os.path.splitext(x)[1].lower() == selected_fileext]
            system.tech_docs.clean_file_paths = [os.path.join(system.tech_docs.clean_path, x) for x in files if os.path.splitext(x)[1].lower() == selected_fileext]
            if len(system.tech_docs.file_paths) > 1:
                system.trainable = False
                print(f"More than one tech support file with the same extension in {system.name}")
        return

    def convert_selected_files(self, file_path: str, section_idx=1, manus_idx=3):
        """
        Convert the selected manuscript files to jsonl format.
        """
        converted_file = run_manus_extractor(file_path, section_col_idx=section_idx, manuscript_col_idx=manus_idx)
        return converted_file  
      
    def copy_files(self):
        """
        Copy the files from the data root to the clean data root.
        """
        for _, system_dict in self.systems.items():
            system: System = system_dict
            # Skip if system is defective / not valid
            if not system.valid:
                continue
            if not system.should_copy:
                print(f"System {system.name} already copied, skipping")
                continue
            # Copy tech docs
            for file, clean_file in zip(system.tech_docs.file_paths, system.tech_docs.clean_file_paths):
                if os.path.exists(clean_file):
                    print(f"File {clean_file} already exists, skipping copy")
                    continue
                shutil.copy(file, clean_file)
            # Copy manuscript
            for file, clean_file in zip(system.manuscript.file_paths, system.manuscript.clean_file_paths):
                if os.path.exists(clean_file):
                    print(f"File {clean_file} already exists, skipping copy")
                    continue
                shutil.copy(file, clean_file)
            # Copy question
            for file, clean_file in zip(system.question.file_paths, system.question.clean_file_paths):
                if os.path.exists(clean_file):
                    print(f"File {clean_file} already exists, skipping copy")
                    continue
                shutil.copy(file, clean_file)
            system.should_copy = False
        return
def remove_faulty_systems(data_conf):
    """
    Remove systems that are faulty.
    """
    filtered_conf = {}
    for system, paths in data_conf.items():
        #Hard coding away systems we don't want to train on
        if "system_20_21_27" in system: # Removing this due to too large source material in generate_training_dataset
            continue
        if "system_80_82_83_84_85" in system:
            continue
        #if "system_50" in system:
        #    continue
        #if "system_70" in system:
        #    continue
        #if "system_77" in system:
         #   continue
        if "system_76" in system:
            continue
        if "system_test" in system:
            continue
        if "system_" in system:
            filtered_conf[system] = paths
    return filtered_conf

def cdata_create_move_paths(data_conf, cdata_path, BASE):
    """
    Create paths for the cleaned data.
    """
    if not os.path.exists(cdata_path):
        os.makedirs(cdata_path)
    for system, paths in data_conf.items():
        system_path = os.path.join(cdata_path, system)
        #Creates root system folder
        if not os.path.exists(system_path):
            os.makedirs(system_path)
        for doc_type, path in paths.items():
            if doc_type == "tech_docs":
                for i in path:
                    old_path = os.path.join(BASE, i)
                    filtered_files = []
                    files = [x for x in os.listdir(old_path)]
                    for file in files:
                        if file.endswith(".docx") or file.endswith(".DOCX"):
                            filtered_files.append(file)
                    for file in filtered_files:
                        shutil.copy(os.path.join(old_path, file), os.path.join(system_path, file) )

            if doc_type == "manuscript":
                for i in path:
                    if not os.path.exists(os.path.join(system_path, os.path.basename(i))):
                        os.makedirs(os.path.join(system_path, os.path.basename(i)))
                    old_path = os.path.join(BASE, i)
                    new_path = os.path.join(system_path, os.path.basename(i))
                    filtered_files = []
                    files = [x for x in os.listdir(old_path)]
                    for file in files:
                        if file.endswith(".docx") or file.endswith(".DOCX") or file.endswith(".jsonl"):
                            filtered_files.append(file)
                    for file in filtered_files:
                        shutil.copy(os.path.join(old_path, file), os.path.join(new_path, file) )

            if doc_type == "question":
                for i in path:
                    if not os.path.exists(os.path.join(system_path, os.path.basename(i))):
                        os.makedirs(os.path.join(system_path, os.path.basename(i)))
                    old_path = os.path.join(BASE, i)
                    new_path = os.path.join(system_path, os.path.basename(i))
                    filtered_files = []
                    files = [x for x in os.listdir(old_path)]
                    for file in files:
                        if file.endswith(".docx") or file.endswith(".DOCX"):
                            filtered_files.append(file)
                    for file in filtered_files:
                        shutil.copy(os.path.join(old_path, file), os.path.join(new_path, file) )

    return


if __name__ == "__main__":
    pass
    cleaner = Cleaner()
    #data_path = os.path.join(BASE, "data")
    #cdata_path = os.path.join(BASE, "data_clean")

    #data_conf = json.load(open(os.path.join(data_path, "data.json"), "r"))

    #filtered_conf = remove_faulty_systems(data_conf)
    #cdata_create_move_paths(filtered_conf, cdata_path)

    #base_path = os.path.join(BASE, DATA_ROOT)
