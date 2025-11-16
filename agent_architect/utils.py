from typing import Any, Dict, List, Optional, Union

from .session_abstraction import AgentSessions


def get_priority_name(queue_name, priority):
    if priority == 0:
        return f"{queue_name}:high"
    else:
        return f"{queue_name}:low"


def transform_priority_name(queue_name: dict, priority: str):
    priority_level = priority.split(":")[-1]
    return queue_name[priority_level]


def get_priority_number(priority: str):
    if priority.endswith("high"):
        return 1
    elif priority.endswith("low"):
        return 2
    elif priority.endswith("triger"):
        return 0
    else:
        return -1


def get_high_low(inputs: List[str]):
    high_priority_channle_name = ""
    low_priority_channle_name = ""
    for i in inputs:
        if i.endswith(":high"):
            high_priority_channle_name = i
        elif i.endswith(":low"):
            low_priority_channle_name = i
    return {"high": high_priority_channle_name, "low": low_priority_channle_name}


def go_next_service(
    current_stage_name: str,
    service_names: Optional[List[str]],
    channels_steps: Optional[Dict[str, List[str]]],
    last_channel: str,
    prioriry: str,
    sid: str = "",
) -> bool:
    """
    Advance the session to the next service in the pipeline.
    Returns True if successfully advanced, False if already at the end.
    """
    if current_stage_name == "start":
        # First real stage is the first service
        if service_names:
            current_stage_name = service_names[0]
            if prioriry not in channels_steps[current_stage_name]:
                return None
            next_channel = f"{current_stage_name}:{prioriry}"
            return next_channel
        else:
            return None

    try:
        idx = service_names.index(current_stage_name)
    except ValueError:
        # Current stage not in pipeline – cannot advance
        return None
    if idx + 1 < len(service_names):
        # Move to next service
        next_service = service_names[idx + 1]
        current_stage_name = next_service
        if prioriry not in channels_steps[current_stage_name]:
            return None

        next_channel = f"{next_service}:{prioriry}"
        return next_channel
    if sid == "" or None:
        return last_channel
    return last_channel + f":{sid}"


def get_all_channels(req: AgentSessions):
    middle_channels = []
    for service in req.service_names:
        if service in req.channels_steps:
            for priority in req.channels_steps[service]:
                middle_channels.append(f"{service}:{priority}")
    return middle_channels + [req.first_channel] + [req.last_channel]
