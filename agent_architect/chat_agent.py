import asyncio
import json
import os
import time
from typing import Dict, List, Any
import redis.asyncio as redis
import logging
from .models_abstraction import AbstractQueueManagerClient, AbstractInferenceClient, DynamicBatchManager
from .session_abstraction import SessionStatus, AgentSessions
from .utils import go_next_service, get_all_channels
from .datatype_abstraction import AudioFeatures
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", 16000))
AGENT_NAME = "chat"
SERVICE_NAMES = ["RAG"]
CHANNEL_STEPS = {"RAG":["high", "low"]}
INPUT_CHANNEL =f"{SERVICE_NAMES[1]}:{CHANNEL_STEPS[SERVICE_NAMES[0]][1]}"
OUTPUT_CHANNEL = f"{AGENT_NAME.lower()}:output"


class RedisQueueManager(AbstractQueueManagerClient):
    """
    Manages Redis-based async queue for inference requests
    """
    def __init__(self, agent_name: str, service_names:str, channels_steps:str, input_channel:str, output_channel:str, redis_url: str = "redis://localhost:6379", timeout:float=30.):
        self.redis_url = redis_url
        self.agent_name = agent_name
        self.service_names = service_names
        self.channels_steps = channels_steps
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.timeout = timeout
        self.redis_client = None
        self.pubsub = None
        self.active_sessions_key = f"{self.agent_name}:active_sessions"
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)
        self.pubsub = self.redis_client.pubsub()
        await self.pubsub.subscribe(self.output_channel)
        logger.info(f"Redis queue manager initialized for queue")
    
    async def start_session(self, sid: str, owner_id:str, kb_ids:List[str], kb_limit:int):
        """Mark a session as active"""
        await self.stop_session(sid)
        agent_session = AgentSessions(sid=sid,
                      agent_name=self.agent_name, 
                      service_names=self.service_names,
                      channels_steps=self.channels_steps,
                      owner_id=owner_id,
                      kb_id=kb_ids,
                      kb_limit=kb_limit,
                      status=SessionStatus.ACTIVE,
                      first_channel=self.input_channel,
                      last_channel=self.output_channel,
                      timeout=self.timeout)
        await self.redis_client.hset(self.active_sessions_key, sid, agent_session.to_json())
        logger.info(f"Session {sid} started")
    
    async def get_status_object(self, sid:str)-> AgentSessions:
        raw = await self.redis_client.hget(self.active_sessions_key, sid)
        if raw is None:
            return None
        return AgentSessions.from_json(raw)
    
    async def cleanup_stopped_sessions(self):
        """Remove all sessions with status 'stop' from active_sessions."""
        sessions = await self.redis_client.hgetall(self.active_sessions_key)
        for sid, raw_status in sessions.items():
            try:
                status_obj = AgentSessions.from_json(raw_status)
                if status_obj.status == SessionStatus.STOP or status_obj.is_expired():
                    print(f"sid {status_obj.sid} is stoped")
                    await self.stop_session(status_obj.sid)
            except Exception as e:
                logger.warning(f"Failed to parse session {sid}: {e}")
                
    async def stop_session(self, sid: str):
        """Mark a session as inactive and cleanup its requests"""
        status_obj = await self.get_status_object(sid=sid)
        deleted_count = await self.redis_client.hdel(self.active_sessions_key, sid)
        if deleted_count > 0:
            await self.cleanup_session_requests(status_obj)
            logger.info(f"Session {sid} stopped (deleted from Redis)")
        else:
            logger.warning(f"Session {sid} was not active - may have been already stopped")
        
    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        status_obj = await self.get_status_object(sid)
        if status_obj is None:
            return False
        if status_obj.is_expired():
            await self.stop_session(sid)
            return False
        return True
        
    async def cleanup_session_requests(self, req: AgentSessions):
        """Remove any queued requests for stopped session"""
        sid = req.sid
        for queue in get_all_channels(req):
            items = await self.redis_client.lrange(queue, 0, -1)
            for item in items:
                try:
                    request_data = json.loads(item)
                    if request_data.get('sid') == sid:
                        await self.redis_client.lrem(queue, 1, item)
                        print(f"remove from channel:{queue}, sid:{sid}")
                except json.JSONDecodeError:
                    continue
        
    async def submit_data_request(self, request_data: AudioFeatures, sid: str) -> str:
        status_obj = await self.get_status_object(sid=sid)
        next_service = go_next_service(current_stage_name="start",
                        service_names=status_obj.service_names,
                        channels_steps=status_obj.channels_steps,
                        last_channel=status_obj.last_channel,
                        prioriry="input")
        await self.redis_client.lpush(next_service, request_data.to_json())
        logger.info(f"Request {sid} submitted to {next_service}")
    
    async def listen_for_result(self, sid: str) -> Dict[str, Any]:
        """Listen for specific request result using a dedicated connection"""
        start_time = time.time()
        status_obj = await self.get_status_object(sid=sid)
        # Create a new Redis client JUST for listening
        temp_client = await redis.from_url(self.redis_url, decode_responses=True)
        pubsub = temp_client.pubsub()
        await pubsub.subscribe(status_obj.last_channel)

        try:
            async for message in pubsub.listen():
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Timeout waiting for result {sid}")
                if message['type'] == 'message':
                    try:
                        result_data = json.loads(message['data'])
                        if result_data['sid'] == sid:
                            return result_data
                    except json.JSONDecodeError:
                        continue
        finally:
            await pubsub.unsubscribe(OUTPUT_CHANNEL)
            await pubsub.close()
            await temp_client.close()
            
    async def close(self):
        """Cleanup resources"""
        if self.pubsub:
            await self.pubsub.unsubscribe(OUTPUT_CHANNEL)
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
    
class InferenceService(AbstractInferenceClient):
    def __init__(
        self,
        agent_name:str,
        service_names:str, 
        channels_steps:str, 
        input_channel:str, 
        output_channel:str,         
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        timeout:float=30.
    ):
        super().__init__()
        self.agent_name = agent_name
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.queue_manager = RedisQueueManager(redis_url=redis_url, agent_name=agent_name, service_names=service_names, channels_steps=channels_steps,input_channel=input_channel, output_channel=output_channel, timeout=timeout)
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)

    async def start_session(self, sid: str, owner_id:str, kb_ids:List[str], kb_limit:int):
        """Mark a session as active"""
        await self.queue_manager.start_session(sid, owner_id, kb_ids, kb_limit)

    async def stop_session(self, sid: str):
        """Stop a specific client session"""
        await self.queue_manager.stop_session(sid)

    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_active(sid)
    
    async def _initialize_components(self):
        await self.queue_manager.initialize()

    async def _cleanup_components(self):
        await self.queue_manager.close()
    
    
    async def _cleanup_stopped_sessions_loop(self):
        """Background loop to clean up stopped sessions."""
        while self.is_running:
            try:
                await self.queue_manager.cleanup_stopped_sessions()
            except Exception as e:
                logger.error(f"Error in stopped session cleanup: {e}")
            await asyncio.sleep(1.0)  # Check every second (adjust as needed)

    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
        asyncio.create_task(self._cleanup_stopped_sessions_loop())
    
    async def predict(self, input_data: bytes, sid: str) -> Any:
        if not await self.is_session_active(sid):
            raise Exception(f"Session {sid} is not active or has been stopped")
        input_request = AudioFeatures(sid=sid, agent_name=self.agent_name, priority=self.input_channel, audio=input_data, sample_rate=SAMPLE_RATE, created_at=None)
        await self.queue_manager.submit_data_request(request_data=input_request, sid=sid)
        
        result_data = await self.queue_manager.listen_for_result(sid)
        # Check if session was stopped during processing
        if not await self.is_session_active(sid):
            raise Exception(f"Session {sid} was stopped during processing")
        
        if result_data.get('error'):
            raise Exception(f"Inference error: {result_data['error']}")
        return result_data['result']
        # return input_request

if __name__ == "__main__":
    import uvicorn

    print("Testing GitHub Actions Docker build...")
    uvicorn.run(
        "call_agent:app",
        host=os.getenv("host", "0.0.0.0"),
        port=os.getenv("port",8000),
        log_level="info",
        reload=True,  # Enable auto-reload during development
    )
