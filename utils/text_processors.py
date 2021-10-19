from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import gym


class TextProcessor(object):
  """docstring for TextProcessor"""
  def __init__(self, max_sent_len):
    super(TextProcessor, self).__init__()
    self.max_sent_len = max_sent_len

# ======================================================
# GPT-2
# ======================================================
class GPT2Processor(TextProcessor):
  """docstring for GPT2Processor"""
  def __init__(self, tokenizer, embedding, **kwargs):
    super(GPT2Processor, self).__init__(**kwargs)
    self.tokenizer = tokenizer
    self.embedding = embedding

  @property
  def vocab_size(self):
    import ipdb; ipdb; ipdb.set_trace()

  def __call__(self, text):
    import ipdb; ipdb.set_trace()


  def gym_observation_space(self):
    import ipdb; ipdb.set_trace()
    return gym.spaces.Box(
      low=0, high=num_possible_tokens, shape=(self.max_sent_len,embedding_size), dtype=np.float32
    )




def load_gpt2_instr_preprocessor(max_sent_len):

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2Model.from_pretrained('gpt2')
  embeddings = model.get_input_embeddings()
  return GPT2Processor(tokenizer, embedding, max_sent_len=max_sent_len)


# ======================================================
# BabyAI
# ======================================================
class BabyAIProcessor(TextProcessor):
  """docstring for GPT2Processor"""
  def __init__(self, instr_preprocessor, **kwargs):
    super(GPT2Processor, self).__init__(**kwargs)
    self.instr_preprocessor = instr_preprocessor

  def __call__(self, text):
    output = np.zeros(self.max_sent_len, dtype=np.int32)

    indices = self.instr_preprocessor([text], torchify=False)[0]
    assert len(indices) < self.max_sent_len, "need to increase sentence length capacity"
    output[:len(indices)] = indices
    return output

  @property
  def vocab_size(self):
    return len(self.instr_preprocessor.vocab.vocab) + 1 #indexing starts at 1


  def gym_observation_space(self):
    num_possible_tokens = len(self.instr_preprocessor.vocab.vocab) + 1 #indexing starts at 1
    assert num_possible_tokens > 0, "vocabulary is empty"
    if num_possible_tokens <= 255:
        mission_dtype=np.uint8
    else:
        mission_dtype=np.int32

    return gym.spaces.Box(
      low=0, high=num_possible_tokens, shape=(self.max_sent_len,num_possible_tokens), dtype=mission_dtype
    )


