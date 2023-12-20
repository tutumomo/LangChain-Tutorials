# %% [markdown]
# # Google Gemini Pro Usage via Gemini API and LangChain
# 
# It's exciting to see, **Gemini Pro** is available via API today.
# 
# Here are some key takeaways for developers:
# 
# - 32K context window for text, and a larger context window to come
# - **free to use right now, within limits**
# - features supported: `function calling`, `embeddings`, `semantic retrieval` and `custom knowledge grounding`, and `chat functionality`
# - supports 38 languages across 180+ countries and territories worldwide
# - Gemini Pro accepts text as input and generates text as output. 
# - A dedicated Gemini Pro Vision multimodal endpoint available today that accepts text and imagery as input, with text output.

# %% [markdown]
# ## Get Your API Key
# 
# Visit [Google AI Studio](https://makersuite.google.com/) to create your *API Key*.

# %% [markdown]
# ## Environment Preparation
# 
# Let's install the required Python packages. If you are not going to use LangChain, you can skip `langchain-google-genai`.

# %%
#pip install -q --upgrade google-generativeai langchain-google-genai python-dotenv

# %% [markdown]
# We could store the Google API Key created in the `.env` file and get it referenced by environmental variable.
# 
# ```shell
# GOOGLE_API_KEY=xxxxxxx
# ```

# %%
from dotenv import load_dotenv
load_dotenv()

# %% [markdown]
# ## Use Google Generative AI SDK to Access Gemini API

# %% [markdown]
# Let's define a helper function `to_markdown` to diplay the model output in a nicer way.

# %%
from IPython.display import display
from IPython.display import Markdown
import textwrap


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# %% [markdown]
# You could refer to the official documentation of the [Generative AI Python SDK](https://ai.google.dev/tutorials/python_quickstart).

# %%
import google.generativeai as genai

# %%
import os
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# %% [markdown]
# ### Text Generation

# %%
model = genai.GenerativeModel(model_name = "gemini-pro")

# %%
prompt_parts = [
    "寫一個Python函數並向我解釋一下",
]

response = model.generate_content(prompt_parts)

# %%
print(response.text)

# %% [markdown]
# ### Image Recognition
# 
# In this section, we will use the image from [Melody Zimmerman](https://unsplash.com/@roseonajourney) - [https://unsplash.com/photos/a-cup-of-coffee-next-to-a-plate-of-food-baNjp1eJAyo](https://unsplash.com/photos/a-cup-of-coffee-next-to-a-plate-of-food-baNjp1eJAyo)

# %%
#! pip install pillow

# %%
import PIL.Image

img = PIL.Image.open('coffee-roll.jpg')
img

# %%
model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(img)

to_markdown(response.text)

# %%
response = model.generate_content(
    [
        "根據這張圖片寫一篇簡短、引人入勝的部落格文章。 它應該包括照片中物體的描述並談論我在東京的旅程。", 
        img
    ], 
    stream=True
)
response.resolve()

# %%
to_markdown(response.text)

# %% [markdown]
# ## Use LangChain to Access Gemini API
# 
# LangChain framework provides a wrapper class **ChatGoogleGenerativeAI** to invoke Gemini API.
# 
# By default, it looks for Google API Key in environmental variable `GOOGLE_API_KEY`.

# %%
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

# %% [markdown]
# #### Safety Settings
# 
# Gemini API enables developers to adjust safety settings on the following 4 dimensions to quickly assess if the application requires more or less restrictive configuration:
# 
# - Harassment
# - Hate speech
# - Sexually explicit
# - Dangerous
# 
# By default, safety settings block content with medium and/or high probability of being unsafe content across all 4 dimensions, which is designed to work for most use cases. Develpers could also adjust its safety settings as needed.
# 
# The probability is rated in general as below:
# 
# - Negligible
# - Low
# - Medium
# - High
# 
# To understand how it's exactly defined in API, please refer to the following documentation:
# 
# - [Harm Category](https://ai.google.dev/api/rest/v1beta/HarmCategory)
# - [Harm Probability](https://ai.google.dev/api/rest/v1beta/SafetyRating#HarmProbability)
# 
# 

# %%
response = model.generate_content('我很憤怒')
response.candidates

# %% [markdown]
# Use `prompt_feedback` attribute of a response to see if it's blocked.

# %%
response.prompt_feedback

# %% [markdown]
# User safety settings in a `generate_content` function call to customize. For example, if you're building a video game dialogue, you may deem it acceptable to allow more content that's rated as dangerous due to the nature of the game.

# %%
safety_settings=[
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    }
]

response = model.generate_content(
    'How are you?', 
    safety_settings=safety_settings
)
response.candidates

# %%
response.prompt_feedback

# %%



