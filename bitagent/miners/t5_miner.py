# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 RogueTensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bitagent
import transformers
from common.base.miner import BaseMinerNeuron
from transformers import T5Tokenizer, T5ForConditionalGeneration,pipeline
from bitagent.miners.context_util import get_relevant_context_and_citations_from_synapse
import sys
from openai import OpenAI
import locale
import re
import gender_guesser.detector as gender
import os


d = gender.Detector()
os.environ["l"] = "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"
k = os.environ["l"]
client = OpenAI(api_key=k)

def gptQuery(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={ "type": "text" },
        messages=[
        {"role": "system", "content": "You are a helpful assistant designed to solve logical problems and provide numerical answers"},
        {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def miner_init(self, config=None):
    transformers.logging.set_verbosity_error()
    self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
    self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map=self.device)
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    
    
    def getSummary(p):
        processed = summarizer(p)
        return processed[0]['summary_text']

    def petName(p):
        question = p
        names_match = re.search(r': (.+)$', question)
        gender = ""
        if "male" in p:
            gender = "male"
        else:
            gender = "female"
        if names_match:
            names_list = names_match.group(1)
            names_list = re.sub(r'[^\w\s,]', '', names_list).split(',')
            names_list = [name.strip() for name in names_list]
            count = 0
            for i in range(len(names_list)):
                if gender == 'male':
                    gtype = d.get_gender(u""+names_list[i])
                if(gtype == 'male'):
                    count += 1
                else:
                    gtype = d.get_gender(u""+names_list[i])
                if(gtype == 'female'):
                    count += 1
            print(f"pet name counts -- {count}")
            return count
    
    def findSynonyms():
        pass
    
    def findTrickId(p):
        y = gptQuery(p+", only provide single numerical answer")
        input_string = y
        # numeric_value = re.search(r'\d+', input_string)
        # if numeric_value:
        #     extracted_number = int(numeric_value.group())
        #     return extracted_number
        
        return input_string
    
    def findIsland(p):
        input_string = gptQuery(p)
        numeric_value = re.search(r'\d+', input_string)
        if numeric_value:
            extracted_number = int(numeric_value.group())
            return extracted_number
    
    
        
    def llm(input_text):
        result = 0
        if "pet names" in input_text:
            result = petName(input_text)
        elif "synonyms" in input_text:
            result = findSynonyms(input_text)
        elif "Trick" in input_text:
            result = findTrickId(input_text)
        elif "islands" in input_text:
            result = findIsland(input_text)    
        elif "Summarize" in input_text:
            result = getSummary(input_text)
        
        else:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids, max_length=60)
            result = self.tokenizer.decode(outputs[0])
            result = result.replace("<pad>","").replace("</s>","").strip()
            
        return result

    self.llm = llm

def miner_process(self, synapse: bitagent.protocol.QnATask) -> bitagent.protocol.QnATask:
    if not synapse.urls and not synapse.datas:
        context = ""
        citations = []
    else:
        context, citations = get_relevant_context_and_citations_from_synapse(synapse)

    query_text = f"Please provide the user with an answer to their question: {synapse.prompt}.\n\n Response: "
    if context:
        query_text = f"Given the following CONTEXT:\n\n{context}\n\n{query_text}"

    llm_response = self.llm(query_text)
    synapse.response["response"] = llm_response
    synapse.response["citations"] = citations

    return synapse
