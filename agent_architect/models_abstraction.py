import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from .datatype_abstraction import Features


class AbstractQueueManager(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    async def initialize(self):
        pass
    
    @abstractmethod
    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        pass
    
    async def cleanup_session_requests(self, sid: str):
        """Remove any queued requests for the sid """
        pass


class AbstractQueueManagerServer(AbstractQueueManager):
    @abstractmethod
    async def get_data_batch(self):
        """Retrieve a batch of audio data from the processing queue."""
        pass
    
    async def is_session_interrupt(self, sid:str):
        """interrupt status for the spacial session id"""
        pass
    
    @abstractmethod
    async def push_result(self):
        """Publish the result of the speech-to-text processing."""
        pass
  
    
class AbstractQueueManagerClient(AbstractQueueManager):
    
    @abstractmethod
    async def start_session(self, sid: str):
        """Start a new client session"""
        pass

    @abstractmethod
    async def stop_session(self, sid: str):
        """Stop a specific client session"""
        pass
    
    @abstractmethod
    async def submit_data_request(self):
        """Submit an audio request to the processing queue."""
        pass
    
    @abstractmethod
    async def listen_for_result(self):
        """Listen for the result of the speech-to-text processing."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the queue manager, releasing any resources."""
        pass
    
        
class AbstractAsyncModelInference(ABC):
    """
    Abstract base class for asynchronous model inference with dynamic batching.
    Subclasses must implement model-specific logic.
    """

    def __init__(self, max_worker: int = 4):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_worker)
        self.stats = {
            'total_batches': 0,
            'total_requests': 0,
            'avg_batch_size': 0,
            'avg_inference_time': 0
        }

    async def process_batch(self, batch: List[Features]) -> Dict[str, Any]:
        """Process a batch of requests asynchronously (final method)."""
        pass

    async def _prepare_batch_inputs(self, batch: List[Features]) -> Any:
        """Prepare inputs for inference. Return format is implementation-defined."""
        pass

    async def _run_async_model_inference(self, prepared_inputs: Any) -> Any:
        """Run async inference (e.g., with vLLM). Must be implemented."""
        pass
    
    def _run_model_inference(self, prepared_inputs: Any) -> Any:
        """Run synchronous inference in thread pool. Must not be async."""
        pass

    async def _process_batch_outputs(self, outputs: Any, batch: List[Features]) -> Dict[str, Any]:
        """Map raw outputs back to request IDs and format results."""
        pass

    async def _handle_batch_error(self, batch: List[Features], error: Exception) -> Dict[str, Any]:
        """Handle errors during batch processing (can be overridden)."""
        error_results = {}
        for request in batch:
            error_results[request.sid] = {
                'result': None,
                'error': str(error)
            }
        return error_results

    def _update_stats(self, batch_size: int, processing_time: float):
        """Update internal statistics (final method)."""
        self.stats['total_batches'] += 1
        self.stats['total_requests'] += batch_size
        self.stats['avg_batch_size'] = (
            self.stats['avg_batch_size'] * (self.stats['total_batches'] - 1) + batch_size
        ) / self.stats['total_batches']
        self.stats['avg_inference_time'] = (
            self.stats['avg_inference_time'] * (self.stats['total_batches'] - 1) + processing_time
        ) / self.stats['total_batches']
  
  
class AbstractInference(ABC):
    """
    Abstract base class for a complete async inference service with dynamic batching,
    queue management, and result publishing.
    """
    @abstractmethod    
    def __init__(self):
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.active_sessions: Dict[str, bool] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
    
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        """Stop the inference service gracefully."""
        pass
    
    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        pass    
    
    @abstractmethod
    async def _initialize_components(self) -> None:
        """Initialize queue manager, inference engine, etc."""
        pass
 
   
class AbstractInferenceServer(AbstractInference):
    """
    Abstract base class for a complete async inference service with dynamic batching,
    queue management, and result publishing.
    """

    @abstractmethod
    async def _process_batches_loop(self) -> None:
        """
        Main loop that:
        - Fetches a batch of requests (with dynamic batching logic)
        - Runs inference
        - Publishes results
        - Updates metrics
        """
        pass


class AbstractInferenceClient(AbstractInference):
    """
    Abstract base class for a complete async inference service with dynamic batching,
    queue management, and result publishing.
    """
    
    async def start_session(self, sid: str):
        """Start a new client session"""
        pass

    
    async def stop_session(self, sid: str):
        """Stop a specific client session"""
        pass
    
    
    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
    
    
    async def stop(self) -> None:
        """Stop the inference service gracefully."""
        self.is_running = False
        if self.processing_task:
            await self.processing_task
        await self._cleanup_components()
    
    
    async def _cleanup_components(self) -> None:
        """Clean up resources (close connections, executors, etc.)."""
        pass

    
    async def predict(
        self,
        input_data: Any,
        sid: str,
        priority: int = 1,
        timeout: float = 30.0
    ) -> Any:
        """
        Public API for submitting a prediction request and awaiting its result.
        
        Parameters:
            input_data: The input payload (e.g., text string).
            sid: Unique session/request ID.
            priority: Request priority (e.g., 0=trigger, 1=high, 2=low).
            timeout: Max time to wait for result (seconds).

        Returns:
            The inference result (e.g., audio array and sample rate).

        Raises:
            Exception: If inference fails or times out.
        """
        pass


class DynamicBatchManager:
    """
    Implements dynamic batching strategies inspired by NVIDIA Triton :cite[2]
    """
    
    def __init__(
        self,
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        preferred_batch_sizes: List[int] = None
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.preferred_batch_sizes = preferred_batch_sizes or [4, 8, 16]
        
        # Metrics
        self.batch_sizes = []
        self.avg_processing_time = 0.0
        
    def should_process_batch(self, current_batch: List[Features], batch_start_time: float) -> bool:
        """Determine if current batch should be processed now"""
        current_size = len(current_batch)
        current_time = time.time()
        batch_age = current_time - batch_start_time
        
        # Check if we reached max batch size :cite[7]
        if current_size >= self.max_batch_size:
            return True
        
        # Check if we reached a preferred batch size :cite[2]
        if current_size in self.preferred_batch_sizes:
            return True
        
        # Check if max wait time reached (delayed batching) :cite[2]
        if batch_age >= self.max_wait_time and current_size > 0:
            return True
        
        # Check for urgent requests (approaching timeout)
        if self._has_urgent_requests(current_batch):
            return True
            
        return False
    
    def _has_urgent_requests(self, batch: List[Features]) -> bool:
        """Check if any requests are approaching timeout"""
        current_time = time.time()
        for request in batch:
            time_until_timeout = request.timeout - (current_time - request.created_at)
            if time_until_timeout < self.max_wait_time:
                return True
        return False
    
    def update_metrics(self, batch_size: int, processing_time: float):
        """Update batch processing metrics"""
        self.batch_sizes.append(batch_size)
        if len(self.batch_sizes) > 100:  # Keep last 100 batches
            self.batch_sizes.pop(0)
        
        # Update running average
        self.avg_processing_time = (
            self.avg_processing_time * (len(self.batch_sizes) - 1) + processing_time
        ) / len(self.batch_sizes)
                       
    