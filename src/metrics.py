import evaluate
from clean_data import Cleaner
from library import LibraryManager
from config import setup_config; setup_config("master")
from llmware.retrieval import Query
import os
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
def load_outputs():
    systems = ["system_21", "system_43fakkel"]

    lib_manager: LibraryManager = LibraryManager(Cleaner())

    libs = {system: lib_manager.libs[system]["manuscript"] for system in lib_manager.libs if system in systems}


    outputs: dict = {}

    for system in systems:
        path = os.path.join("/home/chrislindl/master/output/final_outputs", system)
        files = [x for x in os.listdir(path) if x.endswith(".txt")]
        outputs[system] = {}
        for x in files:
            elements = x.split("_")
            if "manuscript" in elements:
                #output = []
                output = ""
                with open(os.path.join(path, x), "r") as file:
                    for line in file:
                        #output.append(line)
                        line = line.replace("\n", " ")
                        output += line
                outputs[system][elements[1]] = {"output": output}

    for k, lib in libs.items():
        system = outputs[k]
        data = Query(lib_manager.load_lib(lib)).get_whole_library()
        ground_truth = ""
        for x in data:
            if x["content_type"] == "text" or x["content_type"] == "table":
                text: str = x["text"].replace("\n", " ")
                clean_text = " ".join([x for x in text.split() if x != ""])
                ground_truth += clean_text
        for model, input in system.items():
            system[model].update({"ground_truth": ground_truth})
    return outputs
outputs = load_outputs()

print("Computing Bleu Results")
blue_result = bleu_metric.compute(predictions=[outputs["system_21"]["llama"]["output"]],
                                  references=[outputs["system_21"]["llama"]["ground_truth"]])

print("Computing Rouge Results")
rouge_result = rouge_metric.compute(predictions=[outputs["system_21"]["llama"]["output"]],
                                  references=[outputs["system_21"]["llama"]["ground_truth"]])

for system, data in outputs.items():
    for model, content in data.items():
        bleu = bleu_metric.compute(predictions=[content["output"]],
                                   references=[content["ground_truth"]])
        rouge = rouge_metric.compute(predictions=[content["output"]],
                                     references=[content["ground_truth"]])
        print(f"System: {system}, Model: {model}")
        print(f"Bleu; bleu: {bleu["bleu"]:.3f}, precisions: {" ". join([f"{val:.3f}" for val in bleu["precisions"]])}, length_ratio: {bleu["length_ratio"]:.3f}")
        print(f"Rouge; rouge1: {rouge['rouge1']:.3f}, rouge2: {rouge['rouge2']:.3f}, rougeL: {rouge['rougeL']:.3f}, rougeLsum: {rouge['rougeLsum']:.3f}\n")

