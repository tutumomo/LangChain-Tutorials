from dotenv import load_dotenv
load_dotenv()

from IPython.display import display
from IPython.display import Markdown
import textwrap

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

import google.generativeai as genai

import os
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

model = genai.GenerativeModel(model_name = "gemini-pro")

prompt_parts = [
    "寫一個Python函數並向我解釋一下",
]

response = model.generate_content(prompt_parts)

print(response.text)

from langchain_google_genai import ChatGoogleGenerativeAI

# %%
llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("保持健康的最佳做法是什麼？")
to_markdown(result.content)

# %% [markdown]
# ### Advanced Use Cases
# 
# In this section, we are going to cover some interesting use cases of Gemini Pro:
# 
# 1. Chat conversation
# 2. Safety settings

# %% [markdown]
# #### Chat Conversation
# 
# Gemini managed conversations between the user and the model across multiple turns.

# %%
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
chat

# %%
response = chat.send_message("What is mixture of expert model(MOE)? 請用繁體中文回復")
response

# %%
to_markdown(response.text)

# %%
response = chat.send_message("我的第一個問題是甚麼?")
to_markdown(response.text)