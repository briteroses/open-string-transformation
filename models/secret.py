from dataclasses import dataclass
import os

from models.black_box_model import ClaudeFamily


@dataclass
class ClaudeMerlot(ClaudeFamily):
    override_key = os.environ["SECRET_ANTHROPIC_API_KEY"]
    name = os.environ["SECRET_ANTHROPIC_MODEL_NAME"]
    description = "secret"
    scale = None

@dataclass
class ClaudeMalbec(ClaudeFamily):
    override_key = os.environ["SECRET_ANTHROPIC_API_KEY"]
    name = 'research-claude-malbec'
    description = "secret0"
    scale = None