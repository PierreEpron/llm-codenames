from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from transformers import (
    HfArgumentParser,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

from tqdm import tqdm
import numpy as np
import torch

from src.utils import get_hf_token, get_incremented_path
from src import game as G

torch_dtype = torch.bfloat16
device_map = "auto"

@dataclass
class ScriptConfig:
    
    result_path: str = field(metadata={"help":"the path to used to save results."})

    # model
    clm_model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct" , metadata={"help": "the model name"})

    # b&b args
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})

    # prompts

    spy_system_prompt_path: Optional[str] = field(default="data/prompts/spy_system.txt", metadata={"help": "the path containing the prompt"})
    agent_system_prompt_path: Optional[str] = field(default="data/prompts/agent_system.txt", metadata={"help": "the path containing the prompt"})


parser = HfArgumentParser([ScriptConfig, G.GameConfig])
script_config, game_config = parser.parse_args_into_dataclasses()


tokenizer = AutoTokenizer.from_pretrained(script_config.clm_model_name, padding_side="left", token=get_hf_token())
tokenizer.use_default_system_prompt = False
tokenizer.pad_token_id = tokenizer.eos_token_id


if script_config.load_in_8bit or script_config.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_config.load_in_8bit, load_in_4bit=script_config.load_in_4bit
    )
else:
    quantization_config = None


clm_model = AutoModelForCausalLM.from_pretrained(
    script_config.clm_model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    token=get_hf_token()
)

generation_config = GenerationConfig(
  max_length=4096,
  # do_sample=False,
  do_sample=True,
  temperature=.6,
  top_p=.9,
  eos_token_id=tokenizer.eos_token_id,
  pad_token_id=tokenizer.pad_token_id,
)

vocab = Path('data/codenames.txt').read_text().splitlines()
spy_system_prompt = Path(script_config.spy_system_prompt_path).read_text().strip()
agent_system_prompt = Path(script_config.agent_system_prompt_path).read_text().strip()


grid = G.Grid(game_config, vocab)

spy_turns = [
  {'role':'system', 'content':spy_system_prompt + "\n\n" + grid.get_str()}
]

agent_turns = [
  {'role':'system', 'content':agent_system_prompt + "\n\n" + grid.get_str(hide=True)}
]

player_turns = [spy_turns, agent_turns]
player_names = ['spy', 'agent']

result_path = get_incremented_path(script_config.result_path)

torch.cuda.empty_cache()
clm_model.eval()

with torch.no_grad():

    for step in tqdm(list(range(100))):

        current_turns = player_turns[step % 2]

        input_ids = tokenizer.apply_chat_template(current_turns, return_tensors='pt').to(clm_model.device)
        answer = clm_model.generate(input_ids, generation_config)[0, input_ids.shape[1]:]

        for turns in player_turns:
            turns.append({
                'role':'user' if turns == current_turns else 'assistant',
                'content': tokenizer.decode(answer[3:-1]),
            })

        (result_path / f"{player_names[step % 2]}.txt").write_text(tokenizer.decode(tokenizer.apply_chat_template(current_turns)))