import time
from llmware.prompts import Prompt, PromptCatalog
from llmware.util import Utilities, Sources
from llmware.models import ModelCatalog, HFGenerativeModel, GGUFConfigs
import numpy as np
import torch
class CustomInferencer:
    def __init__(self, prompter):
        self.prompter: Prompt = prompter
        modelCatalog = ModelCatalog()
        self.llm_model: HFGenerativeModel = prompter.llm_model
        self.engineered_prompts = []
        #self.llm_model: HFGenerativeModel = modelCatalog.load_model(prompter.model_name)
        pass

    def run(self, prompt_name: str|None = None) -> list:
        prompt = ""
        response_list = []
        response_dict = {}
        for i, batch in enumerate(self.engineered_prompts):
            response_dict = self.prompt_main(batch, prompt_name=prompt_name,
                                            context=self.prompter.source_materials[0]["text"])
            # add details on the source materials to the response dict
            if "metadata" in self.prompter.source_materials[i]:
                response_dict.update({"evidence_metadata": self.prompter.source_materials[i]["metadata"]})

            if "biblio" in self.prompter.source_materials[i]:
                response_dict.update({"biblio": self.prompter.source_materials[i]["biblio"]})

            response_list.append(response_dict)
        return response_list

    def prompt_main (self, prompt, prompt_name=None, context=None, call_back_attempts=1, calling_app_id="",
                     prompt_id=0,batch_id=0, prompt_wrapper=None, promptCard=None,
                     inference_dict=None, max_output=None, temperature=None):

        """ Main inference method to execute inference on loaded model. """
        first_source_only = False

        usage = {}

        if max_output:
            self.llm_max_output_len = max_output

        self.prompter.llm_model.target_requested_output_tokens = self.prompter.llm_max_output_len
        self.prompter.llm_model.add_context = context
        self.prompter.llm_model.add_prompt_engineering = prompt_name


        if True:
            output_dict = self.inference(prompt, inference_dict=inference_dict, prompt_name=prompt_name,
                                                            )

            output = output_dict["llm_response"]

            if isinstance(output,list):
                output = output[0]

            # triage process - if output is ERROR code, then keep trying up to parameter- call_back_attempts
            #   by default - will not attempt to triage, e.g., call_back_attempts = 1
            #   --depending upon the calling function, it can decide the criticality and # of attempts

            if output == "/***ERROR***/":
                # try again
                attempts = 1

                while attempts < call_back_attempts:

                    # wait 5 seconds to try back
                    time.sleep(5)

                    # exact same call to inference
                    output_dict = self.prompter.llm_model.inference(prompt)

                    output = output_dict["llm_response"]
                    # if list output, then take the string from the first output
                    if isinstance(output, list):
                        output = output[0]

                    # keep trying until not ERROR message found
                    if output != "/***ERROR***/":
                        break

                    attempts += 1

                # if could not triage, then present "pretty" error output message
                if output == "/***ERROR***/":
                    if "error_message" in output_dict:
                        output = output_dict["error_message"]
                    else:
                        output = "AI Output Not Available"

        # strip <s> & </s> which are used by some models as end of text marker
        #if not use_fc:
        #    output = str(output).replace("<s>","")
        #    output = str(output).replace("</s>","")

        if "usage" in output_dict:
            usage = output_dict["usage"]

        output_dict = {"llm_response": output, "prompt": prompt,
                       "evidence": context,
                       "instruction": prompt_name, "model": self.llm_model.model_name,
                       "usage": usage,
                       "time_stamp": Utilities().get_current_time_now("%a %b %d %H:%M:%S %Y"),
                       "calling_app_ID": calling_app_id,
                       "rating": "",
                       "account_name": self.prompter.account_name,
                       "prompt_id": prompt_id,
                       "batch_id": batch_id,
                        }

        if context:
            evidence_stop_char = len(context)
        else:
            evidence_stop_char = 0
        output_dict.update({"evidence_metadata": [{"evidence_start_char":0,
                                                   "evidence_stop_char": evidence_stop_char,
                                                   "page_num": "NA",
                                                   "source_name": "NA",
                                                   "doc_id": "NA",
                                                   "block_id": "NA"}]})

        #if register_trx:
        #    self.register_llm_inference(output_dict,prompt_id,trx_dict)

        return output_dict
    def inference(self, prompt, add_context=None, add_prompt_engineering=None, api_key=None,
                  inference_dict=None, prompt_name=None, prompt_wrapper=None):

        """ Executes generation inference on model. """

        self.prompt = prompt

        # first prepare the prompt

        #if add_context:
        #    self.add_context = add_context

        #if add_prompt_engineering:
        #    self.add_prompt_engineering = add_prompt_engineering

        #   add defaults if add_prompt_engineering not set
        #if not self.add_prompt_engineering:

        #    if self.add_context:
        #        self.add_prompt_engineering = "default_with_context"
        #    else:
        #        self.add_prompt_engineering = "default_no_context"

        #   end - defaults update

        #   show warning if function calling model
        #if self.fc_supported:
        #    logger.warning("warning: this is a function calling model - using .inference may lead to unexpected "
        #                    "results.   Recommended to use the .function_call method to ensure correct prompt "
        #                    "template packaging.")

        #if inference_dict:

        #    if "temperature" in inference_dict:
        #        self.temperature = inference_dict["temperature"]

        #    if "max_tokens" in inference_dict:
        #        self.target_requested_output_tokens = inference_dict["max_tokens"]

        #   call to preview (not implemented by default)
        self.llm_model.preview()

        #   START - route to api endpoint
        #if self.api_endpoint:
        #    return self.inference_over_api_endpoint(self.prompt, context=self.add_context,
        #                                            inference_dict=inference_dict)
        #   END - route to api endpoint

        text_prompt = self.prompt

        #promptCard= PromptCatalog().lookup_prompt(prompt_name)
        #prompt_engineered = PromptCatalog().build_core_prompt(promptCard, context=prompt)
        #prompt_engineered = prompt_engineered["core_prompt"]
        #prompt_final = self.wrap_custom(prompt_engineered, self.llm_model.prompt_wrapper)
        #if self.add_prompt_engineering:
        #    prompt_enriched = self.prompt_engineer(self.prompt, self.llm_model.add_context, inference_dict=inference_dict)
        #    prompt_final = prompt_enriched

            # text_prompt = prompt_final + "\n"

            # most models perform better with no trailing space or line-break at the end of prompt
            #   -- in most cases, the trailing space will be ""
            #   -- yi model prefers a trailing "\n"
            #   -- keep as parameterized option to maximize generation performance
            #   -- can be passed either thru model_card or model config from HF

        #text_prompt = prompt_final + self.llm_model.trailing_space

        # second - tokenize to get the input_ids

        tokenizer_output = self.llm_model.tokenizer.encode(text_prompt)
        input_token_len = len(tokenizer_output)
        input_ids = torch.tensor(tokenizer_output).unsqueeze(0)

        #   explicit check and setting to facilitate debugging
        if self.llm_model.use_gpu:
            input_ids = input_ids.to('cuda')
        else:
            input_ids = input_ids.to('cpu')

        # time start
        time_start = time.time()

        #   This simplified greedy sampling generation loop was derived from and inspired by ideas in the
        #   HuggingFace transformers library generation class.
        #   https: //github.com/huggingface/transformers/tree/main/src/transformers/generation
        #   Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc.team, and NVIDIA Corporation.
        #   Licensed under the Apache License, Version 2.0 (the "License")

        # default settings
        pad_token_id = 0

        # for most models, eos_token_id = 0, but llama and mistral = 2
        eos_token_id = [self.llm_model.eos_token_id]
        # eos_token_id = [0]

        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)

        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        new_tokens_generated = 0

        attn_mask = torch.ones(input_ids.shape[1]).unsqueeze(0)

        #   explicit check and setting to facilitate debugging, if needed
        if self.llm_model.use_gpu:
            attn_mask = attn_mask.to('cuda')
        else:
            attn_mask = attn_mask.to('cpu')

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        pkv = None

        # borrow setting from GGUFConfigs
        get_first_token_speed = GGUFConfigs().get_config("get_first_token_speed")
        t_gen_start = time.time()
        first_token_processing_time = -1.0

        while True:

            inp_one_time: torch.LongTensor = input_ids

            if new_tokens_generated > 0:
                inp_one_time = input_ids[:, -1:]

            #   explicit check and setting to facilitate debugging, if needed
            if self.llm_model.use_gpu:
                inp0 = inp_one_time.to('cuda')
                inp1 = attn_mask.to('cuda')
            else:
                inp0 = inp_one_time.to('cpu')
                inp1 = attn_mask.to('cpu')

            # inp3 = torch.LongTensor([new_tokens_generated])

            # need to invoke forward pass on model
            # outputs = self.model(inp0,inp1,pkv)

            #   context manager to avoid saving/computing grads in forward pass
            with torch.no_grad():
                outputs = self.llm_model.model(input_ids=inp0, attention_mask=inp1, past_key_values=pkv,
                                     return_dict=True)

            if new_tokens_generated == 0:
                if get_first_token_speed:
                    first_token_processing_time = time.time() - t_gen_start

            new_tokens_generated += 1

            next_token_logits = outputs.logits[:, -1, :]

            # capture top logits - not currently activated for inference
            # self.register_top_logits(next_token_logits)
            # shape of next_token_logits = torch.Size([1, 32000])
            # logger.debug(f"next token logits shape - {next_token_logits.shape}")

            if self.llm_model.temperature and self.llm_model.sample:
                next_token_scores = next_token_logits / self.llm_model.temperature
            else:
                next_token_scores = next_token_logits

            # get token from logits
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)

            if not self.llm_model.sample:
                # will pull the 'top logit' only
                next_tokens = torch.argmax(probs).unsqueeze(0)
                if len(next_tokens[:, None].shape) == 3:
                    print("Self made, for error checking")
            else:
                # will apply probabilistic sampling
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # new - option to capture logits and output tokens for analysis
            if self.llm_model.get_logits:
                self.llm_model.register_top_logits(next_token_logits)

                # capture the output tokens
                if self.llm_model.use_gpu:
                    next_tokens_np = np.array(next_tokens.to('cpu'))
                else:

                    next_tokens_np = np.array(next_tokens)

                self.llm_model.output_tokens.append(next_tokens_np[0])

            # finished sentences should have their next token be a padding token
            #if len(next_tokens[:, None].shape) == 3:
            #    print("Self made, for error checking")
            #if len(unfinished_sequences.shape) > 1:
            #    if unfinished_sequences.size(dim=1) == 3:
            #        print("Self made, for error checking2")
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            #if len(next_tokens[:, None].shape) == 3:
            #    print("Self made, for error checking")
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            #   testing output in progress starts here
            """
            logger.debug(f"update: input_ids - {input_ids}")
            # outputs_detached = outputs.to('cpu')
            outputs_np = np.array(input_ids[0])
            output_str = self.tokenizer.decode(outputs_np)
            logger.debug(f"update: output string - {output_str}")
            """
            #   end - testing output in progress

            pkv = outputs.past_key_values

            # update attention mask
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((attn_mask.shape[0], 1))], dim=-1)

            # if eos_token was found in one sentence, set sentence to finished
            #if eos_token_id_tensor is not None:
            #    unfinished_sequences = unfinished_sequences.mul(
            #        next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            #    )

            if eos_token_id_tensor is not None:
                # Handle multiple eos_token_ids
                eos_token_id_tensor = eos_token_id_tensor.view(-1, 1)  # Ensure it's a column vector
                eos_mask = next_tokens.unsqueeze(0).ne(eos_token_id_tensor)  # Compare against all eos_token_ids
                unfinished_sequences = unfinished_sequences.mul(eos_mask.prod(dim=0))  # Update unfinished_sequences

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if new_tokens_generated >= self.llm_model.target_requested_output_tokens:
                this_peer_finished = True

            if this_peer_finished:
                break

        #   Generation completed - prepare the output

        if self.llm_model.use_gpu:
            outputs_np = np.array(input_ids[0].to('cpu'))
        else:
            outputs_np = np.array(input_ids[0])

        output_only = outputs_np[input_token_len:]

        output_str = self.llm_model.tokenizer.decode(output_only)

        # post-processing clean-up - stop at endoftext
        eot = output_str.find("<|endoftext|>")
        if eot > -1:
            output_str = output_str[:eot]

        # new post-processing clean-up - stop at </s>
        eots = output_str.find("</s>")
        if eots > -1:
            output_str = output_str[:eots]

        # post-processing clean-up - start after bot wrapper
        bot = output_str.find("<bot>:")
        if bot > -1:
            output_str = output_str[bot + len("<bot>:"):]

        # new post-processing cleanup - skip repeating starting <s>
        boss = output_str.find("<s>")
        if boss > -1:
            output_str = output_str[boss + len("<s>"):]

        # end - post-processing

        total_len = len(outputs_np)

        usage = {"input": input_token_len,
                 "output": total_len - input_token_len,
                 "total": total_len,
                 "metric": "tokens",
                 "processing_time": time.time() - time_start}

        if get_first_token_speed:
            usage.update({"first_token_processing_time": first_token_processing_time})

        output_response = {"llm_response": output_str, "usage": usage}

        if self.llm_model.get_logits:
            output_response.update({"logits": self.llm_model.logits_record})
            output_response.update({"output_tokens": self.llm_model.output_tokens})
            self.llm_model.logits = self.llm_model.logits_record

        # output inference parameters
        self.llm_model.llm_response = output_str
        self.llm_model.usage = usage
        self.llm_model.final_prompt = text_prompt

        self.llm_model.register()

        return output_response
    
    