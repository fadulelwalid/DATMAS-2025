from prompter import Prompter
from pipeline import ModelPipeline
from llmware.prompts import Query, PromptCatalog
from llmware.library import LibraryCatalog, Library
path = "master"
model = ModelPipeline(modelname="qwen", main_path=path)
root_config = model.library_manager.load_data_json("config.json")["llama"]
#llm_config = root_config["llm_config"]
prompter = Prompter(model.cur_model, model.library_manager)
prompter.prepare_prompter(**model.model_configs)
prompter.cur_prompt = "gen_manus_template"

#for lib in 
#query = self.prepare_lib(lib)
libs = LibraryCatalog().all_library_cards()
filtered_libs = []
for element in libs:
    name = element["library_name"]
    if name.endswith("manuscript"):
        filtered_libs.append(element)
examples = []

for idx, lib in enumerate(filtered_libs):
    if idx == 3:
        break
    library = Library().load_library(lib["library_name"])
    q = Query(library).get_whole_library()
    examples.append({"text": "Her kommer et eksempel på et manuskript"})
    for i in q:
        examples.append(i)



#inferencer = prompter.package_query_from_lib(examples)
print("Running inference with examples:")
inferencer = prompter._package_query(examples)
response = inferencer.run()
prompter._package_response(response)
print("Finished inference")
#self.prompter.clear_source_materials()
#return self._package_response(response), response