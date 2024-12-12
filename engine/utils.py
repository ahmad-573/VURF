import os
from PIL import Image
import openai
import numpy as np
import copy

from openai import OpenAI

import requests

from typing import Optional

import fire

from .video_step_interpreters import register_step_interpreters, parse_step

from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM

import sys
sys.path.append("/share/users/ahmad/codellama/")
from llama import Llama




class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    def __init__(self,dataset='nlvr'):
        self.step_interpreters = register_step_interpreters(dataset)

    def execute_step(self,prog_step,inspect):
        step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        return self.step_interpreters[step_name].execute(prog_step,inspect)

    def execute(self,prog,init_state,inspect=False):
        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions]

        html_str = '<hr>'
        for prog_step in prog_steps:
            if inspect:
                step_output, step_html = self.execute_step(prog_step,inspect)
                # print("STEP_HTML: ",step_html)
                # print("HTML_Str: ", html_str)
                # print(type(step_html))
                # print(type(html_str))
                html_str += step_html + '<hr>'
            else:
                step = parse_step(prog_step.prog_str,partial=False)
                step_name = step['step_name']
                if step_name == 'VQA':
                    step_output = ('vqa', step['args']['video'])
                    return step_output, prog.state
                else:
                    step_output = self.execute_step(prog_step,inspect)

        if inspect:
            return step_output, prog.state, html_str

        return step_output, prog.state


class ProgramGenerator():
    def __init__(self,prompter,temperature=0.7,top_p=0.5,prob_agg='mean'):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.prompter = prompter
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg
        # self.generator = Llama.build(
        #     ckpt_dir="/share/users/ahmad/codellama/CodeLlama-7b-Instruct/",
        #     tokenizer_path="/share/users/ahmad/codellama/CodeLlama-7b-Instruct/tokenizer.model",
        #     max_seq_len=2048,
        #     max_batch_size=4,
        # )

        # self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
        # self.model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")

        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen25-7b-multi")

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))

    def generate(self,inputs):
        # response = openai.Completion.create(
        #     model="gpt-3.5-turbo-instruct",
        #     prompt= self.prompter(inputs),
        #     temperature=0.7,
        #     max_tokens=512,
        #     top_p=0.5,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     n=1,
        #     logprobs=1
        # )

        # prob = self.compute_prob(response)
        # prog = response.choices[0]['text'].lstrip('\n').rstrip('\n')
        # return prog, prob
        # client = OpenAI(
        #     api_key = os.getenv("OPENAI_API_KEY"),
        # )

        # completion = client.completions.create( # Change the method
        #     model = "davinci-002",
        #     prompt = self.prompter(inputs),
        #     temperature=0.7,
        #     max_tokens=512,
        #     top_p=0.5,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     n=1
        # )

        # client = OpenAI(
        #     api_key = os.getenv("OPENAI_API_KEY"),
        # )

        # completion = client.completions.create( # Change the method
        #     model = "gpt-3.5-turbo-instruct",
        #     prompt = self.prompter(inputs),
        #     temperature=0.7,
        #     max_tokens=512,
        #     top_p=0.5,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     n=1
        # )


        # prog = completion.choices[0].text.lstrip('\n').rstrip('\n')
        # return prog, 0
        # completion = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "Only output the program in the correct structure as in the examples"},
        #         {"role": "user", "content": self.prompter(inputs)}
        #     ],
        #     temperature=0.7,
        #     max_tokens=512,
        #     top_p=0.5,
        #     frequency_penalty=0,
        #     presence_penalty=0,
        #     n=1
        # )
        # prog = completion.choices[0].message.content.lstrip('\n').rstrip('\n')
        # return prog, 0

        # instructions = [
        #     [
        #         {
        #             "role": "system",
        #             "content": "Only output the program following same structure as the examples"
        #         },
        #         {
        #             "role": "user",
        #             "content": self.prompter(inputs),
        #         }
        #     ]
        # ]

        # results = self.generator.chat_completion(
        #     instructions,  # type: ignore
        #     max_gen_len=None,
        #     temperature=0.2,
        #     top_p=0.95,
        # )

        # prog = results[0]['generation']['content'].lstrip('\n').rstrip('\n').lstrip().rstrip()
        # return prog,0
        
        
        # text = self.prompter(inputs)
        text = "def sum(x,y):"
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        # simply generate a single sequence
        generated_ids = self.model.generate(input_ids, max_length=20)
        result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).lstrip('\n').rstrip('\n').lstrip().rstrip()

        return result, 0



        
    