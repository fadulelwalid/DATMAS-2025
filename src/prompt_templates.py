from llmware.models import PromptCatalog, ModelCatalog

def add_custom_prompt():
    with open("example_manuscript.txt", "r") as f:
        example_manuscript = f.read()

    prompt_list = []
    catalog: PromptCatalog = PromptCatalog()

    run_order = ["blurb1", "$context", "instruction",]
    run_order_qa = ["blurb1", "$context", "instruction",]
    
    run_order4 = ["blurb1", "$context", "instruction"]

    manuscript_prompt_name = "manuscript_tech_docs"
    manuscript_prompt = {
        "blurb1":  
            "Her kommer et teknisk dokument som er unikt og omhandler olje- og gassektoren: ",
        "instruction":
            "Skriv et manuskript basert på det originale tekniske dokumentet. "
            "Manuskriptet skal være tydelig og strukturert. "
            "Manuskriptet skal inneholde hovedtemaet og viktige konsepter rundt dette hovedtemaet. "
            "Husk å ta inn i manuset faglig relevant innhold som er hentet fra det tekniske dokumentet."
            "Manuskriptet skal brukes i e-læring og skal komplementere en opplæringsvideo i form av en fortellertekst. "
            "Forklar viktige begreper på en enkel og forståelig måte. "
            "Unngå punktlister og tabeller. ",
        "system_message":
            "Du er en ansatt i olje- og gassindustrien som jobber med å lage e-læring for ny ansatte. "
            "Din oppgave er å lage et manuskript til en e-læring basert på et teknisk dokument. "
    }
    catalog.add_custom_prompt_card(prompt_name=manuscript_prompt_name, run_order_list=run_order4, prompt_dict=manuscript_prompt)
    prompt_list.append(manuscript_prompt_name)

    manuscript_prompt_name = "manuscript_tech_docs_example"
    manuscript_prompt = {
        "blurb1":  
            "Her kommer et teknisk dokument som er unikt og omhandler olje- og gassektoren: ",
        "instruction":
            "Skriv et manuskript basert på det originale tekniske dokumentet. "
            "Manuskriptet skal være tydelig og strukturert. "
            "Manuskriptet skal inneholde hovedtemaet og viktige konsepter rundt dette hovedtemaet. "
            "Husk å ta inn i manuset faglig relevant innhold som er hentet fra det tekniske dokumentet."
            "Manuskriptet skal brukes i e-læring og skal komplementere en opplæringsvideo i form av en fortellertekst. "
            "Forklar viktige begreper på en enkel og forståelig måte. "
            "Unngå punktlister og tabeller. ",
        "blurb2": "Her er et eksempel på et ferdiglaget manuskript:",
        "example": example_manuscript,
        "system_message":
            "Du er en ansatt i olje- og gassindustrien som jobber med å lage e-læring for ny ansatte. "
            "Din oppgave er å lage et manuskript til en e-læring basert på et teknisk dokument. "
    }
    run_order_example = ["blurb1", "$context", "instruction", "blurb2", "example"]
    catalog.add_custom_prompt_card(prompt_name=manuscript_prompt_name, run_order_list=run_order_example, prompt_dict=manuscript_prompt)
    prompt_list.append(manuscript_prompt_name)

    manusscript_prompt_instr_first_name = "manuscript_tech_docs_instr_first"
    manuscript_prompt_instr_first = {
        "instruction":
            "Skriv et manuskript basert på det kommende originale tekniske dokumentet. "
            "Manuskriptet skal være tydelig og strukturert. "
            "Manuskriptet skal inneholde hovedtemaet og viktige konsepter rundt dette hovedtemaet. "
            "Husk å ta inn i manuset faglig relevant innhold som du henter fra det tekniske dokumentet."
            "Manuskriptet skal brukes til e-læring og skal brukes til en opplæringsvideo. "
            "Skriv manuskriptet som om en forteller skal lese opp manuskriptet. "
            "Unngå punktlister og tabeller. ",
        "example": f"Her er en mal for hvordan et manuskript skal se ut: {example_manuscript}",
        "blurb1":  
            "Her kommer det tekniske dokumentet som er unikt og omhandler olje- og gassektoren: ",
        "blurb2": "Skriv nå et manuskript.",
        "system_message":
            "Du er en ansatt i olje- og gassindustrien som jobber med å lage e-læring for ny ansatte. "
            "Din oppgave er å lage et manuskript til en e-læring basert på et teknisk dokument. "
    }
    # Two versions of the run order, one with example and one without
    #run_order_instr_first = ["instruction", "blurb1", "$context", "blurb2"]
    run_order_instr_first = ["instruction", "example", "blurb1", "$context", "blurb2"]
    catalog.add_custom_prompt_card(prompt_name=manusscript_prompt_instr_first_name, run_order_list=run_order_instr_first, prompt_dict=manuscript_prompt_instr_first)
    prompt_list.append(manusscript_prompt_instr_first_name)

    mcq_prompt_name = "generate_mcqs_tech_docs"
    mcq_prompt = {
        "blurb1": (
            "Les nøye gjennom følgende manuskript, som basereres seg på et teknisk dokument som er unikt og og omhandler olje- og gassektoren. "
            "Her kommer manuskriptet:"
        ),
        "instruction": (
            "Generer 10 flevalgsoppgaver basert på det oppgitte manuskriptet. "
            "Spørsmålene må være skreddersydd til det spesifikke innholdet. "
            "Spørsmålene må være faglig relevante og nøyaktige. "
            "Hver oppgave skal ha fire svaralternativer: ett riktig og tre plausibelt feilaktige svar. "
            "De feilaktige svarene skal være realistiske og kun relateres til det unike manuskriptet eller olje- og gassindustrien, men tydelig feil ved grundig lesing. "
            "Bruk et profesjonelt og klart språk i både spørsmålene og svaralternativene. "
            "Vanskelighetsgraden skal være passende for fagpersoner innen bransjen."
            "Gi ut det riktige svaret i egen linje etter spørsmålene. Ikke gi noe forklaring på svaret."
            "Formater hver oppgave slik:\n\n"
            "Spørsmål: [Spørsmål]\n"
            "A) [Plausibelt feil svar]\n"
            "B) [Plausibelt feil svar]\n"
            "C) [Riktig svar]\n"
            "D) [Plausibelt feil svar]\n\n"
            "Riktig svar: C)\n\n"
        ),
        "system_message": (
            "Du er en ekspert på olje- og gassektoren og generering av faglige spørsmål. "
            "Svarene dine er presise, faktabaserte og fri for unødvendige detaljer. "
            "Både spørsmålene og svarene skal være på norsk og faglig korrekt innen olje- og gassindustrien."
        )
    }

    #test2 = PromptCatalog().add_custom_prompt_card(prompt_name=mcq_prompt_name, run_order_list=run_order, prompt_dict=mcq_prompt)
    catalog.add_custom_prompt_card(prompt_name=mcq_prompt_name, run_order_list=run_order_qa, prompt_dict=mcq_prompt)
    prompt_list.append(mcq_prompt_name)

    training_prompt_name = "training_general"
    training_prompt = {
        "blurb1": "Du er teknkisk skribent og har fått i oppgave å lage e-læring og kurs for olje- og gassindustrien. "
                    "Du skal lage et manuskript til en opplæringsvideo. Samtidig skal du lage vilkårlige passende oppgaver til kurset eksplisitt basert på manuskriptet."
                    "Du blir trent nå på å lage et manuskript basert på et teknisk dokument."
                    "Du blir også trent på å lage oppgaver basert på manuset der oppgavene er avkryssing og feil svar skal være plausible.",
        "instruction": "Basert på enten det tekniske dokumentet eller manuset, generer respektivt manuskriptet eller oppgaver.",
        "system_message":
            "Du er en ekspert på olje- og gassektoren og generering av faglige spørsmål. "
            "Svarene dine er presise, faktabaserte og fri for unødvendige detaljer. "
            "Både spørsmålene og svarene skal være på norsk og faglig korrekt innen olje- og gassindustrien."
    }
    catalog.add_custom_prompt_card(prompt_name=training_prompt_name, run_order_list=run_order, prompt_dict=training_prompt)
    prompt_list.append(training_prompt_name)

    template_manuscript_name = "gen_manus_template"
    template_manuscript = {
        "blurb1": (
            "Les nøye gjennom følgende flere manuskripter, som basereres seg på et teknisk dokument som er unikt og og omhandler olje- og gassektoren. "
        ),
        "instruction": (
            "Generer en mal for hvordan et manuskript skal se ut. "
            "Baser malen på fellestrekk fra eksempel manuskriptene. "
            "Hold samme struktur som eksempel manuskriptene. "
        ),
        "blurb2": "Skriv nå en mal.",
        "system_message": (
            "Du er en ekspert på olje- og gassektoren og generering av faglige spørsmål. "
            "Svarene dine er presise, faktabaserte og fri for unødvendige detaljer. "
            "Både spørsmålene og svarene skal være på norsk og faglig korrekt innen olje- og gassindustrien."
        )

    }
    run_order_template = ["instruction", "blurb1", "$context", "blurb2"]
    catalog.add_custom_prompt_card(prompt_name=template_manuscript_name, run_order_list=run_order_template, prompt_dict=template_manuscript)
    prompt_list.append(template_manuscript_name)

    manuscript_summary_prompt_name = "manuscript_summary_tech_docs"
    manuscript_prompt = {
        "blurb1": (
            "Du skal nå bruke følgende strukturerte sammendrag som en referanse for å skrive et manuskript basert på "
            "det originale tekniske dokumentet. Sammendraget inneholder hovedtemaene og viktige konsepter som må reflekteres i manuset. "
            "Manuset skal brukes i e-læring - enten som tekstinnhold eller som grunnlag for en opplæringsvideo."
        ),
        "blurb2": "Her kommer det originale dokumentet:",
        "instruction": (
            "Analyser det originale dokumentet med utgangspunkt i sammendraget. "
            "Skriv et tydelig, pedagogisk og strukturert manuskript som følger hovedtemaene i logisk rekkefølge. "
            "Manuset skal egne seg for e-læring og kunne brukes enten som ren tekst eller som fortellertekst i en opplæringsvideo. "
            "Bruk tydelige underoverskrifter, korte og konsise avsnitt. Forklar viktige begreper på en enkel og forståelig måte. "
            "Unngå punktlister og tabeller, og skriv i stedet med en naturlig og sammenhengende flyt. "
            "Ikke gjenta sammendraget direkte – bruk det som en veiviser til hvilke temaer som skal dekkes."
        ),
        "system_message": (
            "Du er en erfaren teknisk skribent. Du bruker sammendraget som tematisk veiledning og henter utfyllende informasjon fra det originale dokumentet. "
            "Du skriver på en måte som gjør det lett å forstå, med fokus på å forklare ting tydelig og strukturert."
            "Resultatet skal være en klar og sammenhengende tekst som forklarer tekniske konsepter på en pedagogisk måte."
        )
    }
    catalog.add_custom_prompt_card(prompt_name=manuscript_summary_prompt_name, run_order_list=run_order, prompt_dict=manuscript_prompt)
    prompt_list.append(manuscript_summary_prompt_name)

    return prompt_list




def add_models(selected_model: dict):
    model_card_list = []
    ModelCatalog().register_new_finetune_wrapper("qwen_wrapper",
                                                 system_start="<|im_start|>system",
                                                 system_stop="<|im_end|>\n",
                                                 main_start="<|im_start|>user",
                                                 main_stop="<|im_end|>\n",
                                                 llm_start="<|im_start|>assistant")
    
    ModelCatalog().register_new_finetune_wrapper(name="glm_wrapper",
                                                 system_start="[gMASK]<sop><|system|>\n",
                                                 system_stop="<|endoftext|>\n",
                                                 main_start="<|user|>\n",
                                                 main_stop="<|endoftext|>\n",
                                                 llm_start="<|assistant|>\n")
    
    ModelCatalog().register_new_finetune_wrapper("llama_wrapper", ### If the backslash n change affects model performance, thans why for this model
                                                 system_start="<|start_header_id|>system\n\n",
                                                 system_stop="<|end_header_id|>",
                                                 main_start="<|start_header_id|>user\n\n",
                                                 main_stop="<|end_header_id|>",
                                                 llm_start="<|start_header_id|>assistant\n\n")
    
    ModelCatalog().register_new_finetune_wrapper(name="llamaV2_wrapper",
                                                 main_start="<|start_header_id|>user<|end_header_id|>\n\n",
                                                 main_stop="<|eot_id|>",
                                                 llm_start="<|start_header_id|>assistant<|end_header_id|>\n\n",
                                                 system_start="<|start_header_id|>system<|end_header_id|>\n\n",
                                                 system_stop="<|eot_id|>")


    #Very important! We set instruction_following to true!
    #Possibility for context length of 32768 according to ollama
    def generate_model_card(model_name: str, context_window: str, prompt_wrapper: str, display_name: str, custom_model_repo: str):
        return {
            "model_name": model_name,
            "context_window": context_window,
            "prompt_wrapper": prompt_wrapper,
            "hf_repo": custom_model_repo,
            "display_name": display_name,
            "temperature": 0.3,
            "trailing_space": "",
            "model_family": "HFGenerativeModel",
            "model_category": "generative_local",
            "model_location": "hf_repo",
            "instruction_following": True,
            "link": f"https://huggingface.co/{model_name}",
            "custom_model_files": [],
            "custom_model_repo": custom_model_repo
        }
    
    if selected_model:
        config = selected_model["config"]
        prompt_wrapper: str = config["prompt_wrapper"]
        display_name: str = config["display_name"]
        model_name: str = config["model_name"]
        custom_model_repos: str = config["model_repo_root"]
        context_window: str = config["context_window"]
    
        model_card = generate_model_card(model_name=model_name,
                                        prompt_wrapper=prompt_wrapper,
                                        display_name=display_name,
                                        custom_model_repo=custom_model_repos,
                                        context_window=context_window)
        ModelCatalog().register_new_model_card(model_card)
        model_card_list.append(model_name)
        return
    model_card_qwen =   {"model_name": "qwen",
                        "context_window": 32768, #burde klare 128k også --> https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
                        "prompt_wrapper": "qwen_wrapper",

                        # hf_model_name should correspond to the hf repo/model standard
                        "hf_repo": "model/qwen/qwen-model/base",
                        "display_name": "qwen", "temperature": 0.3, "trailing_space": "",
                        "model_family": "HFGenerativeModel", "model_category": "generative_local",
                        "model_location": "hf_repo", "instruction_following": True,
                        "link": "https://huggingface.co/Qwen",
                        "custom_model_files": [], "custom_model_repo": "model/qwen/qwen-model/base"}
    
    model_card_glm =    {"model_name": "glm",
                        "context_window": 131072,
                        "prompt_wrapper": "glm_wrapper",
                        "hf_repo": "glm", #Might change this according to qwen hf_repo
                        "display_name": "glm", "temperature": 0.3, "trailing_space": "",
                        "model_family": "HFGenerativeModel", "model_category": "generative_local",
                        "model_location": "hf_repo", "instruction_following": True,
                        "link": "https://huggingface.co/THUDM",
                        "custom_model_files": [], "custom_model_repo": "model/glm/glm-model/base"}
    model_card_deepseek = {"model_name": "deepseek",
                      "context_window": 32768, #hvorfor ikke 128k? --> https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
                      "prompt_wrapper": "deepseek_wrapper",

                      # hf_model_name should correspond to the hf repo/model standard
                      "hf_repo": "model/deepseek/deepseek-model/base",
                      "display_name": "deepseek", "temperature": 0.3, "trailing_space": "",
                      "model_family": "HFGenerativeModel", "model_category": "generative_local",
                      "model_location": "hf_repo", "instruction_following": True,
                      "link": "https://huggingface.co/deepseek-ai",
                      "custom_model_files": [], "custom_model_repo": "model/deepseek/deepseek-model/base"}
    
    model_card_llama = {"model_name": "llama",
                        "context_window": 128000, #8k dersom quanitized --> https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
                        "prompt_wrapper": "llama_wrapper",
                        
                        "hf_repo": "/mnt/k/Models/llama/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
                        "display_name": "llama", "temperature": 0.3, "trailing_space": "",
                        "model_family": "HFGenerativeModel", "model_category": "generative_local",
                        "model_location": "hf_repo", "instruction_following": True,
                        "link": "https://huggingface.co/meta-llama",
                        "custom_model_files": [], "custom_model_repo": "/mnt/k/Models/llama/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"}

    #ModelCatalog().register_new_hf_generative_model()
    #ModelCatalog().register_new_model_card(model_card_qwen)
    #ModelCatalog().register_new_model_card(model_card_deepseek)
    #ModelCatalog().register_new_model_card(model_card_llama)
    return
