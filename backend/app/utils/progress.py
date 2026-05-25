import threading
from typing import Dict, List, Optional

class ProgressStore:
    def __init__(self):
        self._store: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
        
    def add_node(self, thread_id: str, node_name: str) -> None:
        """Add a completed/running node to the thread's progress."""
        with self._lock:
            if thread_id not in self._store:
                self._store[thread_id] = []
            if node_name not in self._store[thread_id]:
                self._store[thread_id].append(node_name)
                
    def get_progress(self, thread_id: str) -> List[str]:
        """Get the current progress (list of node names) for a thread."""
        with self._lock:
            return list(self._store.get(thread_id, []))
            
    def clear_progress(self, thread_id: str) -> None:
        """Clear progress for a completed thread."""
        with self._lock:
            if thread_id in self._store:
                del self._store[thread_id]

# Global singleton
progress_store = ProgressStore()

def set_progress(thread_id: str, node_name: str) -> None:
    progress_store.add_node(thread_id, node_name)

def get_progress(thread_id: str) -> List[str]:
    return progress_store.get_progress(thread_id)

def clear_progress(thread_id: str) -> None:
    progress_store.clear_progress(thread_id)
