import json
import time
from dataclasses import asdict, dataclass, field
from typing import List, Union


@dataclass
class Features:
    sid: str
    agent_name:str
    priority:str
    created_at: float
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.timeout

    def to_json(self) -> str:
        """Convert instance to JSON string"""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create instance from JSON string"""
        data = json.loads(json_str)
        return cls(**data)
    
    def refresh_time(self):
        self.created_at = time.time()


@dataclass
class AudioFeatures(Features):
    audio:bytes
    sample_rate:int
    
    
@dataclass
class TextFeatures(Features):
    text:str


@dataclass
class RAGFeatures(TextFeatures):
    owner_id:str
    kb_id:List[str]
    kb_limit:int


@dataclass
class ChannelNames:
    input_channel:Union[str, List[str]]
    output_channel:Union[str, List[str]]
    channel_names:List[str] = field(default_factory=list)
    
    def get_all_channels(self):
        if isinstance(self.input_channel, str):
            input_channel = [self.input_channel]
        else:
            input_channel = self.input_channel
        if isinstance(self.output_channel, str):
            output_channel = [self.output_channel]
        else:
            output_channel = self.output_channel
        return input_channel + output_channel + self.channel_names
    