from .request_openai_utils import (
    OpenAIModel,
)
from .utils import (
    extract_largest_json,
    convert_set_to_list,
    reformat_objective_facts
)

import os
from openai import OpenAI

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
API_KEY = os.getenv("API_KEY", "None")
MODEL_NAME = os.getenv("MODEL_NAME", None)
if CUSTOM_CORPUS_HOME == None:
    raise EnvironmentError("CUSTOM_CORPUS_HOME environment variable is not set")
if MODEL_NAME == None:
    raise EnvironmentError("MODEL_NAME environment variable is not set")
BASE_URL = os.getenv("BASE_URL", None)
if BASE_URL == None:
    CLIENT = OpenAI(api_key=API_KEY)
else:
    CLIENT = OpenAI(base_url=f"{BASE_URL}", api_key=API_KEY)

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", None)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", None)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", None)
if EMBEDDING_BASE_URL == None:
    EMBEDDING_CLINET = OpenAI(
        api_key=EMBEDDING_API_KEY
    )
else:
    EMBEDDING_CLINET = OpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_API_KEY
    )