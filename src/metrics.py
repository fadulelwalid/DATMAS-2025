import evaluate
from evaluate.visualization import radar_plot
from clean_data import Cleaner
from library import LibraryManager

from llmware.retrieval import Query
import os
import matplotlib.pyplot as plt
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def load_outputs(output_type: str = "manuscript", output_string: str = "manuscript", results_path: str = "output/final_outputs"):
    models = MODELS
    systems = SYSTEMS
    lib_manager: LibraryManager = LibraryManager(Cleaner())

    libs = {system: lib_manager.libs[system][output_type] for system in lib_manager.libs if system in systems}


    outputs: dict = {}

    for system in systems:
        path = os.path.join(results_path, system)
        files = [x for x in os.listdir(path) if x.endswith(".txt")]
        outputs[system] = {}
        for x in files:
            elements = x.split("_")
            if output_string in elements:
                #output = []
                output = ""
                for model in models:
                    if model in x:
                        model_name = model
                with open(os.path.join(path, x), "r") as file:
                    for line in file:
                        #output.append(line)
                        line = line.replace("\n", " ")
                        output += line
                outputs[system][model_name] = {"output": output}

    for k, lib in libs.items():
        system = outputs[k]
        data = Query(lib_manager.load_lib(lib)).get_whole_library()
        ground_truth = ""
        for x in data:
            x.update({"text": x["text_search"]})
            if x["content_type"] == "text" or x["content_type"] == "table":
                text: str = x["text"].replace("\n", " ")
                clean_text = " ".join([x for x in text.split() if x != ""])
                ground_truth += clean_text
        for model, input in system.items():
            system[model].update({"ground_truth": ground_truth})
    inverted = {
    model: {example: models[model] for example, models in outputs.items() if model in models}
    for model in {m for models in outputs.values() for m in models}
    }
    return outputs, inverted


def compute_metrics(outputs):
    bleu_raw = []
    rouge_raw = []
    model_names = []

    systems = []
    latex_bleu_string = []
    latex_bleu_string.append(f"\tSystems & Bleu & Precisions & Length Ratio \\\\")
    latex_rouge_string = []
    latex_rouge_string.append(f"\tSystems & Rouge1 & Rouge2 & RougeL & RougeLsum \\\\")
    for model, data in outputs.items():
        bleu_val = []
        precisions = []
        length_ratios = []
        rouge1 = []
        rouge2 = []
        rougeL = []
        rougeLsum = []

        model_names.append(model)
        for system_name, content in data.items():
            bleu = bleu_metric.compute(predictions=[content["output"]],
                                    references=[content["ground_truth"]])
            rouge = rouge_metric.compute(predictions=[content["output"]],
                                        references=[content["ground_truth"]])
            
            bleu_raw.append(bleu)
            rouge_raw.append(rouge)


            bleu_val.append(bleu["bleu"])
            precisions.append(bleu["precisions"])
            length_ratios.append(bleu["length_ratio"])

            rouge1.append(rouge["rouge1"])
            rouge2.append(rouge["rouge2"])
            rougeL.append(rouge["rougeL"])
            rougeLsum.append(rouge["rougeLsum"])

            #print(f"System: {system}, Model: {model}")
            #print(f"Bleu; bleu: {bleu["bleu"]:.3f}, precisions: {" ". join([f"{val:.3f}" for val in bleu["precisions"]])}, length_ratio: {bleu["length_ratio"]:.3f}")
            #print(f"Rouge; rouge1: {rouge['rouge1']:.3f}, rouge2: {rouge['rouge2']:.3f}, rougeL: {rouge['rougeL']:.3f}, rougeLsum: {rouge['rougeLsum']:.3f}\n")
        bleu_metric_avg = sum(bleu_val) / len(bleu_val)
        precisions_avg = [sum(x) / len(x) for x in zip(*precisions)]
        length_ratio_avg = sum(length_ratios) / len(length_ratios)
        rouge1_avg = sum(rouge1) / len(rouge1)
        rouge2_avg = sum(rouge2) / len(rouge2)
        rougeL_avg = sum(rougeL) / len(rougeL)
        rougeLsum_avg = sum(rougeLsum) / len(rougeLsum)
        #systems.append({
        #    "system": system,
        #    "bleu": bleu_metric_avg,
        #    "precisions": precisions_avg,
        #    "length_ratio": length_ratio_avg,
        #    "rouge1": rouge1_avg,
        #    "rouge2": rouge2_avg,
        #    "rougeL": rougeL_avg,
        #    "rougeLsum": rougeLsum_avg
        #})
        system = {
            "system": system_name,
            "bleu": f"{bleu_metric_avg:.3f}",
            "precisions": " ".join([f"{val:.3f}" for val in precisions_avg]),
            "length_ratio": f"{length_ratio_avg:.3f}",
            "rouge1": f"{rouge1_avg:.3f}",
            "rouge2": f"{rouge2_avg:.3f}",
            "rougeL": f"{rougeL_avg:.3f}",
            "rougeLsum": f"{rougeLsum_avg:.3f}"
        }
        systems.append(system)
        #latex_bleu_string.append(f"{system_name.replace("_", "\\_")} & {bleu_metric_avg:.3f} & {' '.join([f'{val:.3f}' for val in precisions_avg])} & {length_ratio_avg:.3f} & {rouge1_avg:.3f} & {rouge2_avg:.3f} & {rougeL_avg:.3f} & {rougeLsum_avg:.3f} \\\\")
        latex_bleu_string.append(f"\t{system["system"].replace('_', '\\_')} & {system["bleu"]} & {system["precisions"]} & {system["length_ratio"]} \\\\")
        latex_rouge_string.append(f"\t{system["system"].replace('_', '\\_')} & {rouge1_avg:.3f} & {rouge2_avg:.3f} & {rougeL_avg:.3f} & {rougeLsum_avg:.3f} \\\\")
    for i in latex_bleu_string:
        print(i)
    print("\n")
    for i in latex_rouge_string:
        print(i)
    print("\n")

    return systems, bleu_raw, rouge_raw, model_names

def generate_average_metrics(bleu_list: list, rouge_list: list):
    bleu_val = []
    precisions = []
    word_pair = []
    length_ratios = []
    rouge1 = []
    rouge2 = []
    rougeL = []
    rougeLsum = []
    
    for bleu, rouge in zip(bleu_list, rouge_list):
        bleu_val.append(bleu["bleu"])
        precisions.append(bleu["precisions"][0])
        word_pair.append(bleu["precisions"][1])
        length_ratios.append(bleu["length_ratio"])

        rouge1.append(rouge["rouge1"])
        rouge2.append(rouge["rouge2"])
        rougeL.append(rouge["rougeL"])
        rougeLsum.append(rouge["rougeLsum"])


    bleu_metric_avg = sum(bleu_val) / len(bleu_val)
    precisions_avg = sum(precisions) / len(precisions)
    word_pair_avg = sum(word_pair) / len(word_pair)
    length_ratio_avg = sum(length_ratios) / len(length_ratios)
    rouge1_avg = sum(rouge1) / len(rouge1)
    rouge2_avg = sum(rouge2) / len(rouge2)
    rougeL_avg = sum(rougeL) / len(rougeL)
    rougeLsum_avg = sum(rougeLsum) / len(rougeLsum)
    return {
        "bleu": bleu_metric_avg,
        "precisions": precisions_avg,
        "word pair": word_pair_avg,
        #"length_ratio": length_ratio_avg,
        "rouge1": rouge1_avg,
        "rouge2": rouge2_avg,
        "rougeL": rougeL_avg,
    }

def compute_model_metrics(models, outputs):

    model_metrics = []
    model_names = []
    raw_model_metrics = []
    #for model, data in outputs.items():
    for model in MODELS:
        data = outputs[model]
        model_names.append(model)
        bleu_raw = []
        rouge_raw = []
        raw_systems_metrics = []
        #for system_name, content in data.items():
        for system_name in SYSTEMS:
            content = data[system_name]
            combined_raw = {}
            bleu = bleu_metric.compute(predictions=[content["output"]],
                                    references=[content["ground_truth"]])
            rouge = rouge_metric.compute(predictions=[content["output"]],
                                        references=[content["ground_truth"]])
            combined_raw.update(bleu)
            combined_raw.update(rouge)
            combined_raw.update({"precisions": bleu["precisions"][0]})
            combined_raw.update({"word_pair": bleu["precisions"][1]})
            bleu_raw.append(bleu)
            rouge_raw.append(rouge)
            raw_systems_metrics.append(combined_raw)
        metrics = generate_average_metrics(bleu_raw, rouge_raw)
        model_metrics.append(metrics)
        raw_model_metrics.append(raw_systems_metrics)
    return model_metrics, model_names, raw_model_metrics

def print_latex_table(metrics):
    for idx, model in enumerate(metrics):
        print(f"Metrics for model: {MODELS[idx]}")
        latex_row = []
        latex_row.append(f"\tSystems & Bleu & Precisions & Word Pair & Length Ratio & Rouge1 & Rouge2 & RougeL \\\\")
        for index, data in enumerate(model):
            latex_string = f"\tSystem {index+1} & {data['bleu']:.3f} & {data['precisions']:.3f} & {data['word_pair']:.3f} & {data['length_ratio']:.3f} & {data['rouge1']:.3f} & {data['rouge2']:.3f} & {data['rougeL']:.3f} \\\\"
            latex_row.append(latex_string)
        for i in latex_row:
            print(i)
        print("\n\n")

def get_bleu_rouge_metrics(results_path: str = "output/final_outputs"):
    # Load outputs for manuscript and MCQ
    outputs_manuscript, model_outputs_manuscript = load_outputs(results_path=results_path)
    outputs_mcqs, model_outputs_mcqs = load_outputs(output_type="question", output_string="mcqs", results_path=results_path)

    # Compute metrics for manuscript and MCQ outputs
    metrics_manuscript, model_names, manuscript_raw_metrics = compute_model_metrics(MODELS, model_outputs_manuscript)
    metrics_mcqs, model_names, mcqs_raw_metrics = compute_model_metrics(MODELS, model_outputs_mcqs)

    print_latex_table(manuscript_raw_metrics)
    print_latex_table(mcqs_raw_metrics)
    plot_manuscript = radar_plot(data=metrics_manuscript, model_names=model_names)
    plot_manuscript.savefig("maunscript_metrics.png", bbox_inches='tight')

    plot_manuscript = radar_plot(data=metrics_mcqs, model_names=model_names)
    plot_manuscript.savefig("mcqs_metrics.png", bbox_inches='tight')
    return

def get_num_success_plot():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    token_sizes = ["11.4K", "17.9K", "15.6K", "9K", "3.5K", "7.4K", "7.2K", "2.5K", "7.7K", "6.6K"]
    x_names = [f"System No. {i}\n{token}" for i, token in zip(x, token_sizes)]
    y_names = ["Qwen", "Qwen Train", "Qwen1M", "Llama", "Llama Train"]
    y = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 4, 0, 1, 0, 1, 4],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 2, 4, 9, 1, 1, 1, 0, 14, 2]]
    plt.figure(figsize=(10, 6))
    for idx, model_scores in enumerate(y):
        plt.plot(x, model_scores, marker='o', label=y_names[idx])

    plt.xticks(x, x_names, rotation=45)
    plt.xlabel("Systems")
    plt.ylabel("Failure Count")
    plt.title("Number of Failures per System for Each Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("num_failure_plot_token.png")
    plt.show()

def load_msqc_outputs(modelname: str, lib_manager: LibraryManager):
    base_path = "output/final_outputs"
    system_outputs = {}
    system_answers = {}
    for system in SYSTEMS:
        path = os.path.join(base_path, system)
        mcqs = [x for x in os.listdir(path) if x.endswith(".txt") and "mcqs" in x]
        manuscript = Query(lib_manager.load_lib(lib_manager.libs[system]["manuscript"])).get_whole_library()
        manus = []
        for i in manuscript:
            if i["content_type"] == "text":
                manus.append(i)
        mcq = []
        for model in MODELS:
            if model != modelname: continue
            for filename in mcqs:
                file = filename.split("_system_")[0]
                file = file.split("_")[1:]
                file = "_".join(file)
                if modelname == file:
                    mcq.append(filename)
                    break

        output = []
        answer = []
        for x in manus:
            line = x["text"]
            output.append({"text": line})
        output.append({"text": "\n\n"})
        output.append({"text": "Her kommer spørsmålene:\n"})
        for x in mcq:
            with open(os.path.join(path, x), "r") as file:
                for line in file:
                    if "Riktig svar" in line:
                        answer.append(line)
                        output.append({"text": "Riktig svar: "})
                    else:
                        output.append({"text": line})

        system_outputs.update({system: output})
        system_answers.update({system: answer})
    return system_outputs, system_answers

def get_mcqs_answers():
    import json
    for model in MODELS:
        _, answers = load_msqc_outputs(model, LibraryManager(Cleaner()))
        output_path = f"mcqs_answers_{model}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in answers.items():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

def evaluate_msqc():
    from pipeline import ModelPipeline
    for model in ["qwen"]:
        pipeline = ModelPipeline(model, "master")   
        inputs, _ = load_msqc_outputs(model, pipeline.library_manager)
        prompter = pipeline.prompter
        prompter.set_prompt("eval_mcqs_answers")

        prompter.prepare_prompter(**pipeline.model_configs)
        print(f"Running evaluation for model: {model}")
        for system, input in inputs.items():
            prompter.cur_lib = system
            print(f"Running evaluation for system: {system}")
            prompter.run_stage_two(input, raw=True)

def print_mcqs_results():
    x_axis = ["Qwen", "Qwen Train", "Qwen1M", "Llama", "Llama Train"]
    y_axis_fault = [35, 59, 2, 11, 36]
    y_axis_correct = [18, 16, 67, 52, 31]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_axis, y_axis_fault, color='tomato')
    plt.ylim(0, 100)
    plt.xlabel("Model")
    plt.ylabel("Failure Rate (%)")
    plt.title("Failure Rate per Model for MCQ Generation")
    plt.tight_layout()
    plt.savefig("mcqs_failure_rate.png")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x_axis, y_axis_correct, color='mediumseagreen')
    plt.ylim(0, 100)
    plt.xlabel("Model")
    plt.ylabel("Similar Answers (%)")
    plt.title("Similar Answers between MCQ Generation and Answering")
    plt.tight_layout()
    plt.savefig("mcqs_similar_answers.png")

MODELS = ["llama", "llama_train", "qwen", "qwen_train", "qwen1m"]

SYSTEMS = []

if __name__ == "__main__":
    root_dir = "master"
    from config import setup_config; setup_config(root_dir)
    results_path = "output/final_outputs"
    get_bleu_rouge_metrics(results_path=results_path)
    #get_num_success_plot()
    #evaluate_msqc()
    #get_mcqs_answers()
    print_mcqs_results()
    pass
