"""
  -*- encoding: utf-8 -*-
  @Author: zhaojingtong
  @Time  : 2024/11/28 19:16
  @Email: 2665109868@qq.com
  @function
"""





from transformers import BertModel, BertTokenizer
import re

model_path = '/data/zhaojingtong/Biobert_embedding/biobert'
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False )
model = BertModel.from_pretrained(model_path)
equence_Example = "PHYHIP"
encoded_input = tokenizer(equence_Example, return_tensors='pt')
output = model(**encoded_input)
print(output['last_hidden_state'][0][0].shape)

