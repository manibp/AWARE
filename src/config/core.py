from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, PositiveInt
from strictyaml import YAML, load


# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "models"

class AppConfig(BaseModel):
    ''' Application config'''

    package_name: str
    country_dict: str
    language_dict: str
    ctry_lang_map: str
    country: str
    languages: List[str]
    ss_threshold: float
    name: str
    location: str
    add_keywords: str
    custom_search_query: str
    pinecone_index: str
    embed_model_name: str
    embed_model_dims: PositiveInt
    generative_model_name: str
    max_tokens: PositiveInt
    llm: str
    model: str
    temperature: float
    chunk_size: PositiveInt
    context_window: PositiveInt
    num_output: PositiveInt
    chunk_overlap_ratio: int
    separator: str

def find_config() -> Path:
    """Locate the configuration file."""

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config file not found at {CONFIG_FILE_PATH}")

def fetch_config_from_yaml(cfg_path:Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config()
    
    if cfg_path:
        with open(cfg_path, 'r') as config_path:
            parsed_config =load(config_path.read())
            return parsed_config
    raise OSError(f"No config file found at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> AppConfig:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = AppConfig(**parsed_config.data)

    return _config

config = create_and_validate_config()

