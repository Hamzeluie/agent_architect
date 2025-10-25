import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SessionStatus:
    ACTIVE = "active"
    INTERRUPT = "interrupt"
    STOP = "stop"


class AgentSessions:
    def __init__(
        self,
        sid: str,
        agent_name: str,
        service_names: Optional[List[str]],
        channels_steps: Optional[Dict[str, List[str]]],
        owner_id:str,
        kb_id:List[str]=[None],
        kb_limit:int=5,
        status: SessionStatus = SessionStatus.ACTIVE,
        timeout: float = 30.0,
        first_channel:str=None,
        last_channel:str=None,
        created_at: Optional[float] = None,
    ):
        self.sid = sid
        self.status = status
        self.timeout = timeout
        self.created_at = created_at if created_at is not None else time.time()
        self.agent_name = agent_name
        self.first_channel = first_channel
        self.last_channel = last_channel
        self.service_names = service_names or []
        self.channels_steps = OrderedDict(channels_steps or {})
        self.owner_id = owner_id
        self.kb_id = kb_id
        self.kb_limit = kb_limit

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
            
    def refresh_time(self) -> None:
        """Update the creation timestamp to extend session lifetime."""
        self.created_at = time.time()

    def is_expired(self) -> bool:
        """Check if the session has exceeded its timeout."""
        return (time.time() - self.created_at) > self.timeout        

    def to_json(self) -> str:
        """Serialize session state to JSON string."""
        data = {
            "sid": self.sid,
            "status": self.status,
            "timeout": self.timeout,
            "created_at": self.created_at,
            "agent_name": self.agent_name,
            "first_channel": self.first_channel,
            "last_channel": self.last_channel,
            "service_names": self.service_names,
            "owner_id" : self.owner_id,
            "kb_id" : self.kb_id,
            "kb_limit" : self.kb_limit,
            "channels_steps": dict(self.channels_steps),  # OrderedDict → dict for JSON
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentSessions":
        """Deserialize session state from JSON string."""
        data = json.loads(json_str)
        # Reconstruct OrderedDict for channels_steps
        channels_steps = OrderedDict(data.get("channels_steps", {}))
        return cls(
            sid=data["sid"],
            status=data["status"],
            timeout=data["timeout"],
            created_at=data["created_at"],
            agent_name=data["agent_name"],
            first_channel=data["first_channel"],
            last_channel=data["last_channel"],
            service_names=data["service_names"],
            owner_id=data["owner_id"],
            kb_id=data["kb_id"],
            kb_limit=data["kb_limit"],
            channels_steps=channels_steps,
        )

    def __repr__(self) -> str:
        return (f"AgentSessions(sid={self.sid}, agent={self.agent_name}, status={self.status}, create_at={self.created_at}, first_channel={self.first_channel}, last_channel={self.last_channel}, owner_id: {self.owner_id}") 
    
