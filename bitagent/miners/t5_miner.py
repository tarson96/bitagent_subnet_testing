import bitagent
import transformers
from common.base.miner import BaseMinerNeuron
from transformers import T5Tokenizer, T5ForConditionalGeneration
from bitagent.miners.context_util import get_relevant_context_and_citations_from_synapse
from transformers import AutoTokenizer, AutoModelForCausalLM
import bittensor as bt
import torch
from transformers import pipeline
# from Model_Test_Miner import Json_To_Vector ,Response
from Miner_Response import Miner_Model
def miner_init(self, config=None):
    transformers.logging.set_verbosity_error()
    # self.tokenizer =  AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta",legacy=False)
    # self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map=self.device)
    # pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
    # self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
    # self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map=self.device)
    # self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/zephyr-7B-beta-GPTQ", device_map=self.device)
    # self.model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GPTQ", device_map=self.device)
    bt.logging.debug("Loading model Zephyr 1111")
    
def llm(input_text):
    result = Miner_Model(input_text)
    bt.logging.info(f"Response From Zephyr : >>>>>>>>{result}")
    bt.logging.info("Loading model Zephyr 3333")
    # response is typically: <pad> text</s>
    # result = result.replace("<pad>","").replace("</s>","").strip()
    return result

def miner_process(self, synapse: bitagent.protocol.QnATask) -> bitagent.protocol.QnATask:
    
    if not synapse.urls and not synapse.datas:
        context = ""
        citations = []
    else:
        context, citations = get_relevant_context_and_citations_from_synapse(synapse)
    # y = "metaphoric phrases"
    # x = f"Need a random word that has really good synonyms.  Do not provide the synonyms, just provide the random word that has good, clear synonyms. Random word:{y} "
    query_text = f"Please provide the user with an answer to their question: {synapse.prompt}.\n\n Response: "
    if context:
        query_text = f"Given the following CONTEXT:\n\n{context}\n\n{query_text}"
    bt.logging.info("Loading model Zephyr 2222")
    # query_text = f"Resolve this :{x} in proper statement"
    llm_response = self.llm(query_text)
    bt.logging.info(f"Response From Zephyr : >>>>>>>>{llm_response}")
    synapse.response["response"] = llm_response
    synapse.response["citations"] = citations
    print("response raw ---->   ", llm_response)
    return synapse