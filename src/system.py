class System:
    def __init__(self, name, path, clean_path):
        self.name = name
        self.path = path
        self.clean_path = clean_path

        self.valid: bool = True
        self.trainable: bool = True
        self.should_copy: bool = False
        self.tech_docs: DataType = None
        self.manuscript: DataType = None
        self.question: DataType = None

class DataType:
    def __init__(self, type: str, path: str, clean_path, files: list):
        self.type: str = type
        self.path: str = path
        self.clean_path: str = clean_path
        self.files: list = files
        self.clean_files: list = []
        self.filetype: str = ""
        self.file_paths: list = []
        self.clean_file_paths: list = []

        self.converted: bool = False
        self.converted_files: list = []
        self.converted_file_paths: list = []
        self.files_to_copy: list = []