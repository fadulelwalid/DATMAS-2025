from llmware.configs import LLMWareConfig, PostgresConfig
import os
from prompt_templates import add_custom_prompt, add_models
def setup_config(path_from_home: str):
    LLMWareConfig().set_home(os.path.join(os.environ.get("HOME"), path_from_home))
    LLMWareConfig().set_active_db("postgres")
    LLMWareConfig().set_vector_db("postgres")
    PostgresConfig().set_config("user_name", "postgres")
    PostgresConfig().set_config("pw", "root")
    return

def setup_models(selected_model: dict):
    model_list = add_models(selected_model)
    return model_list