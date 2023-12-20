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
#pip install --upgrade google-generativeai langchain-google-genai python-dotenv

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
    "寫一個 Python 的函式，然後向初學者解釋函式的用法，請使用繁體中文",
]

response = model.generate_content(prompt_parts)

# %%
print(response.text)

# %% [markdown]
# ### Image Recognition
# 
# In this section, we will use the image from [Melody Zimmerman](https://unsplash.com/@roseonajourney) - [https://unsplash.com/photos/a-cup-of-coffee-next-to-a-plate-of-food-baNjp1eJAyo](https://unsplash.com/photos/a-cup-of-coffee-next-to-a-plate-of-food-baNjp1eJAyo)

# %%
#pip install pillow

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
        "根據這張圖片寫一篇簡短的、引人入勝的部落格文章。 它應該包括照片中物體的描述並談論我在東京的旅程", 
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


