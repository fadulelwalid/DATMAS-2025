from llmware.prompts import Prompt, Sources
from llmware.models import PromptCatalog
from library import LibraryManager
import json
from llmware.retrieval import Query
from prompter import instantiate_tokenizer, Prompter
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
import transformers
import os

class Trainer:
    def __init__(self, config: dict, model_name: str, library: LibraryManager, prompts: list, want_text_only: bool = False):
        model_config = config[f"{model_name}"]
        self.model_name = model_config["config"]["model_name"]
        self.generation_config = config.get("llm_config", {})
        self.saved_models_path = model_config["config"]["train_path"]
        self.saved_checkpoint = os.listdir(model_config["config"]["train_path"])
        self.tokenizer = instantiate_tokenizer(model_config["config"]["tokenizer"])
        self.cur_model = model_config["config"]["model_repo_root"]
        self.cur_prompt: str = ""
        self.library: LibraryManager = library
        self.libs = self.library.libs
        self.prompts = prompts

        self.prompter: Prompter = Prompter(model_config, library)

        self.datasets_path = model_config["config"]["training_datasets"]
        #self.excluded_libs = ["system_50", "system_70", "system_77"]
        self.excluded_libs = []
        self.datasets = os.listdir(self.datasets_path)
        self.cur_dataset: str|None = None # Expects to be a name for external naming sakes.
        self.tokens_count: list = []

        self.want_text_only = want_text_only # All libraries it extracts from are text part only, no tables.


    def get_cur_dataset(self):
        # Should return filename with extension
        # Setting of cur_dataset does not use extension
        if self.cur_dataset is not None:
            return f"{self.cur_dataset}.jsonl" if ".jsonl" not in self.cur_dataset else self.cur_dataset

            return self.cur_dataset
        cur_dataset = None
        for dataset in self.datasets: # Iterate over datasets in dataset folder
            # Skip if prompt name is not part of filename
            if self.cur_prompt[0] not in dataset: continue
            if self.cur_prompt[1] not in dataset: continue
            if self.model_name not in dataset: continue
            cur_dataset = dataset
        if cur_dataset is None:
            return None
        return cur_dataset
    
    def set_prompts(self, tech_prompt: str, manus_prompt: str):
        if tech_prompt in self.prompts and manus_prompt in self.prompts:
            self.cur_prompt = (tech_prompt, manus_prompt)
        else:
            raise ValueError("Prompts not found")
        return

    def run_generate(self):

        prompt_names: tuple = self.cur_prompt
        training_examples = self.generate_training_dataset(self.libs, prompt_names, only_text=False)
        savename = f"{self.cur_dataset}.jsonl" if self.cur_dataset is not None else f"training_dataset_{self.model_name}_{'---'.join(prompt_names)}.jsonl"
        save_training_dataset(training_examples, os.path.join(self.datasets_path, savename))
        self.datasets.append(savename)  # Add to datasets list
        self.save_final_token_count()
        self.tokens_count = []  # Reset token count after saving


    def _apply_engineered_prompt(self, prompt, promptCard):
        prompt_engineered = self.prompter.build_core_prompt(promptCard, context=prompt)
        return prompt_engineered["core_prompt"]

    def _get_engineered_prompt(self, lib, prompter: Prompter, promptCard, only_text: bool = False):
        prompt = get_prompt(self, lib, prompter, only_text=self.want_text_only)
        if prompt is None: return None
        prompt_engineered = self.prompter.build_core_prompt(promptCard, context=prompt)
        return prompt_engineered["core_prompt"]

    def generate_training_dataset(self, systems_dict: dict, prompt_names: tuple, only_text: bool = False):
        self.prompter.prepare_prompter(**self.generation_config)
        prompter = self.prompter.prompter
        # prompter = Prompt(tokenizer=self.tokenizer).load_model(self.model_name)
        tech_promptCard = PromptCatalog().lookup_prompt(prompt_names[0])
        manus_promptCard = PromptCatalog().lookup_prompt(prompt_names[1])
        training_examples = []
        train_examples_count = 0
        for system_name, system_dict in systems_dict.items():
            print(f"Processing system: {system_name}")
            #if system_name != "system_62": continue
            if system_name in self.excluded_libs: 
                continue
            #Loading libraries for this given system
            lib_tech = self.library.load_lib(system_dict["tech_docs"])
            lib_manuscript = self.library.load_lib(system_dict["manuscript"])
            lib_question = self.library.load_lib(system_dict["question"])

            #Creates training example piped from technical documents to manuscript
            tech_prompt = self._get_engineered_prompt(lib_tech, prompter, tech_promptCard)
            if tech_prompt is None: print(f"Skipping system {system_name} due to missing prompt");continue
            manuscript_answer = get_prompt(self, lib_manuscript, prompter, only_text=self.want_text_only)
            if manuscript_answer is None: print(f"Skipping system {system_name} due to missing prompt or answer");continue
            training_examples.append({"prompt": tech_prompt, "completion": manuscript_answer})

            #Creates training example piped from manuscript to question
            manuscript_prompt = self._apply_engineered_prompt(manuscript_answer, manus_promptCard)
            question_answer = get_prompt(self, lib_question, prompter, only_text=self.want_text_only)
            if question_answer is None: print(f"Skipping system {system_name} due to missing prompt or answer");continue
            training_examples.append({"prompt": manuscript_prompt, "completion": question_answer})
            train_examples_count += 1
        print(f"Total training examples: {train_examples_count}")
        return training_examples
    
    def set_train_name(self):
        return self.get_cur_dataset().replace(".jsonl", "").replace("training_dataset_", "")
    
    def run_training(self):
        # Remember to set prompt name before running this function
        if self.cur_prompt == "":
            raise ValueError("Prompt not set")
        # We prefer own specified dataset over the one in the folder
        dataset = self.cur_dataset if self.cur_dataset is not None else self.get_cur_dataset()
        if dataset is None:
            raise ValueError("No dataset found")

        training_dataset = load_dataset("json", data_files=os.path.join(self.datasets_path, self.get_cur_dataset()), split="train")

        output_dir = os.path.join(self.saved_models_path)
        model_save = os.path.join(self.saved_models_path)
        print(f" Saving checkpoints to {output_dir}")
        print(f" Saving model to {model_save}")
        model = transformers.AutoModelForCausalLM.from_pretrained(self.cur_model,
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                )
        training_arg = SFTConfig(output_dir=output_dir, per_device_eval_batch_size=1,
        #training_arg = SFTConfig(output_dir="/mnt/k/Models/llama_train", per_device_eval_batch_size=1,
                                per_device_train_batch_size=1, num_train_epochs=25, model_init_kwargs={"low_cpu_mem_usage": True, })
        #peft_conf = LoraConfig(lora_alpha = 16, lora_dropout = 0.1, r = 64, task_type="CAUSAL_LM")
        trainer: SFTTrainer = SFTTrainer(
            model,
            #config=config,
            #peft_config=peft_conf,
            train_dataset=training_dataset,
            args=training_arg,
        )
        trainer.train()
        model_to_save = trainer.model
        #trainer.log_metrics()
        #trainer.save_metrics()

        model_to_save.save_pretrained(model_save)
        #model_to_save.save_pretrained("/mnt/k/Models/llama_train1")
    def save_final_token_count(self):
        folder = os.path.join(self.saved_models_path, self.set_train_name())
        if not os.path.exists(folder):
            print(f" Creating folder {folder} for token stats")
            os.makedirs(folder)
        print(f" Saving token stats to {folder}")
        json.dump(self.tokens_count, open(os.path.join(folder, "token_stats.json"), "w"), indent=4)

    def save_token_stats(self, lib_name: str, tokens: int):
        # Save token stats to class variable
        # This is used to save the token stats to a json file
        self.tokens_count.append({"lib_name": lib_name, "tokens": tokens})

    def instantiate_tokenizer(tokenizer_path: str):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
            torch_dtype="auto",)
        return tokenizer



def get_prompt(obj: Trainer, lib, prompter: Prompt, only_text: bool = False):
    sources: Sources = Sources(prompter)
    q = Query(lib).get_whole_library()
    lib_name = lib.library_name
    filtered_q = []
    if only_text:
        filtered_q = [x for x in q if x["content_type"] == "text"]
    else:
        filtered_q = q
        
    sources.package_source(filtered_q)
    if len(prompter.source_materials) == 0:
        print(f" No source material found for library: {lib_name}")
        return None
        #raise ValueError("No source material found")
    tot_tokens = 0
    if len(prompter.source_materials) > 1:
        for i in prompter.source_materials:
            tot_tokens += Sources(prompter).token_counter(i["text"])
        obj.save_token_stats(lib_name, tot_tokens)  
        print(f" More than one source material found. Skipping library: {lib_name}, total tokens: {tot_tokens}")
        return None
        #raise ValueError(f"More than one source material found. Skipping library: {lib}, total tokens: {tot_tokens}")
    print(f" Training on lib: {lib_name}. Tokens for this library: {sources.token_counter(prompter.source_materials[0]['text'])}")
    tot_tokens += Sources(prompter).token_counter(prompter.source_materials[0]["text"])
    obj.save_token_stats(lib_name, tot_tokens)  
    context = prompter.source_materials[0]["text"]
    prompter.clear_source_materials()
    return context

def apply_engineered_prompt(prompt, promptCard):
    prompt_engineered = PromptCatalog().build_core_prompt(promptCard, context=prompt)
    return prompt_engineered["core_prompt"]

def get_engineered_prompt(lib, prompter: Prompt, promptCard):
    prompt = get_prompt(lib, prompter, promptCard)
    prompter.build_core_prompt(promptCard, context=prompt["text"])
    prompt_engineered = PromptCatalog().build_core_prompt(promptCard, context=prompt)
    return prompt_engineered["core_prompt"]

def generate_training_dataset(systems_dict: dict, tokenizer_name: str):
    tokenizer = instantiate_tokenizer(tokenizer_name)
    prompter = Prompt(tokenizer=tokenizer).load_model("qwen")
    #prompter = Prompt(tokenizer=tokenizer).load_model("llama")
    promptCard = PromptCatalog().lookup_prompt("summarize_topics_tech_docs")
    #libs = json.load(open("data_clean/libs.json", "r"))
    #lib = load_lib(libs["system_62"]["tech_docs"])
    #prompt = get_engineered_prompt(lib, prompter, promptCard)
    #completion = get_prompt(lib, prompter, promptCard)
    #prompt_engineered = PromptCatalog().apply_prompt_wrapper(prompt_engineered, prompter.prompt_wrapper, instruction=promptCard["system_message"])
    training_examples = []
    for system_name, system_dict in systems_dict.items():
        print(f"Processing system: {system_name}")
        #if system_name != "system_62": continue

        #Loading libraries for this given system
        lib_tech = load_lib(system_dict["tech_docs"])
        lib_manuscript = load_lib(system_dict["manuscript"])
        lib_question = load_lib(system_dict["question"])

        #Creates training example piped from technical documents to manuscript
        tech_prompt = get_engineered_prompt(lib_tech, prompter, promptCard)
        manuscript_answer = get_prompt(lib_manuscript, prompter, promptCard)
        training_examples.append({"prompt": tech_prompt, "completion": manuscript_answer})


        #Creates training example piped from manuscript to question
        manuscript_prompt = apply_engineered_prompt(manuscript_answer, promptCard)
        question_answer = get_prompt(lib_question, prompter, promptCard)
        training_examples.append({"prompt": manuscript_prompt, "completion": question_answer})
    return training_examples
def save_training_dataset(training_examples, fname):
    with open(fname, "w") as f:
        for example in training_examples:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")

def run_training():
    training_dataset = load_dataset("json", data_files="training_dataset.jsonl", split="train")
    #training_args = TrainingArguments(
    #    output_dir="outputs",
    #    per_device_train_batch_size=1,
    #    gradient_accumulation_steps=4,
    #    num_train_epochs=2,
    #    logging_steps=10,
    #    save_steps=200,
    #    evaluation_strategy="steps",
    #    eval_steps=500,
    #    save_total_limit=2,
    #)

    #model = transformers.AutoModelForCausalLM.from_pretrained("/mnt/k/Models/llama/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
    model = transformers.AutoModelForCausalLM.from_pretrained("model/qwen/qwen-model/base",
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            )
    training_arg = SFTConfig(output_dir="/mnt/k/Models/qwen_train", per_device_eval_batch_size=1,
    #training_arg = SFTConfig(output_dir="/mnt/k/Models/llama_train", per_device_eval_batch_size=1,
                             per_device_train_batch_size=1, num_train_epochs=5, model_init_kwargs={"low_cpu_mem_usage": True, })
    #peft_conf = LoraConfig(lora_alpha = 16, lora_dropout = 0.1, r = 64, task_type="CAUSAL_LM")
    trainer = SFTTrainer(
        model,
        #config=config,
        #peft_config=peft_conf,
        train_dataset=training_dataset,
        args=training_arg,
    )
    trainer.train()
    model_to_save = trainer.model
    model_to_save.save_pretrained("/mnt/k/Models/qwen_train1")
    #model_to_save.save_pretrained("/mnt/k/Models/llama_train1")

def run_generate():
    model_conf = json.load(open("config.json", "r"))
    tokenizer_name = model_conf["qwen"]["local"]["base"]
    #tokenizer_name = model_conf["llama"]["local"]["base"]

    libs = json.load(open("data_clean/libs.json", "r"))

    training_examples = generate_training_dataset(libs, tokenizer_name)
    save_training_dataset(training_examples, "training_dataset.jsonl")
    #save_training_dataset(training_examples, "training_dataset_llama.jsonl")



if __name__ == "__main__":
    # pip install trl
    from config import setup_config
    model_conf = json.load(open("config.json", "r"))
    selected_model = model_conf["qwen"]
    #selected_model = model_conf["llama"]
    setup_config("master", selected_model)

    run_training()
    #run_generate()



    #run_training()
