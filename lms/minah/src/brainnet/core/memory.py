"""
Mem0 integration with pgvector for persistent cross-session memory

Uses PostgreSQL with pgvector extension for high-performance vector similarity search.
Falls back to SQLite for simple storage when PostgreSQL is not available.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, Any, List
from pathlib import Path

from .memory_milvus import MilvusMemoryManager

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

# Check pgvector availability
PGVECTOR_AVAILABLE = False
try:
    import psycopg2
    PGVECTOR_AVAILABLE = True
except ImportError:
    pass

class MemoryManager:
    """
    Memory manager using Mem0 with pgvector for persistent cross-session memory.
    
    Uses PostgreSQL with pgvector extension for:
    - High-performance vector similarity search
    - ACID compliance for trade data integrity
    - Scalable storage for large memory sets
    
    Falls back to SQLite if PostgreSQL/pgvector is not available.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        user_id: str = "trader",
        session_id: Optional[str] = None,
    ):
        self.config = config or {}
        self.user_id = user_id
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize Mem0 or fallback
        self.mem0_client = None
        self.use_fallback = False
        self.pgvector_conn = None

        if MEM0_AVAILABLE:
            try:
                self._init_mem0()
            except Exception as e:
                print(f"Mem0 initialization failed, using SQLite fallback: {e}")
                self.use_fallback = True
        else:
            print("Mem0 not available, using SQLite fallback")
            self.use_fallback = True

        if self.use_fallback:
            self._init_sqlite_fallback()

        self.milvus_manager = MilvusMemoryManager(self.config) if self.config.get('use_milvus', False) else None

    def _init_mem0(self):
        """Initialize Mem0 client with pgvector backend."""
        mem0_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "phi3.5",
                    "api_key": self.config.get("llm_api_key", "ollama"),
                    "openai_base_url": self.config.get("llm_base_url", "http://localhost:11434/v1"),
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "nomic-embed-text",
                    "api_key": "ollama",
                    "openai_base_url": self.config.get("llm_base_url", "http://localhost:11434/v1"),
                }
            }
        }

        # Get PostgreSQL connection string
        postgres_url = self.config.get("postgres_url", "")
        
        # If no explicit postgres_url, try to construct from environment
        if not postgres_url:
            pg_host = os.getenv("PGVECTOR_HOST", "localhost")
            pg_port = os.getenv("PGVECTOR_PORT", "5432")
            pg_user = os.getenv("PGVECTOR_USER", "postgres")
            pg_pass = os.getenv("PGVECTOR_PASSWORD", "")
            pg_db = os.getenv("PGVECTOR_DB", "brainnet")
            
            if pg_pass:
                postgres_url = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
            else:
                postgres_url = f"postgresql://{pg_user}@{pg_host}:{pg_port}/{pg_db}"

        # Configure pgvector as the vector store
        # pgvector is preferred for:
        # - Better query performance with IVFFlat/HNSW indexes
        # - ACID compliance for financial data
        # - Native PostgreSQL integration
        mem0_config["vector_store"] = {
            "provider": "pgvector",
            "config": {
                "connection_string": postgres_url,
                "collection_name": self.config.get("pgvector_collection", "brainnet_memories"),
                "embedding_model_dims": self.config.get("embedding_dims", 768),
            }
        }

        # Initialize pgvector extension if we have direct connection
        if PGVECTOR_AVAILABLE and postgres_url:
            try:
                self._init_pgvector_extension(postgres_url)
            except Exception as e:
                print(f"pgvector extension setup warning: {e}")

        self.mem0_client = Memory.from_config(mem0_config)
    
    def _init_pgvector_extension(self, connection_string: str):
        """
        Initialize pgvector extension and create necessary indexes.
        
        Creates the vector extension and optimized indexes for
        high-performance similarity search.
        """
        import psycopg2
        
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        cursor = conn.cursor()
        
        try:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create optimized index for similarity search (if table exists)
            # HNSW index provides better query performance for high-dimensional vectors
            cursor.execute("""
                DO $$
                BEGIN
                    IF EXISTS (SELECT FROM information_schema.tables 
                               WHERE table_name = 'brainnet_memories') THEN
                        -- Create HNSW index for fast approximate nearest neighbor search
                        IF NOT EXISTS (SELECT 1 FROM pg_indexes 
                                      WHERE indexname = 'brainnet_memories_embedding_idx') THEN
                            CREATE INDEX brainnet_memories_embedding_idx 
                            ON brainnet_memories 
                            USING hnsw (embedding vector_cosine_ops)
                            WITH (m = 16, ef_construction = 64);
                        END IF;
                    END IF;
                END $$;
            """)
        finally:
            cursor.close()
            conn.close()

    def _init_sqlite_fallback(self):
        """Initialize SQLite fallback storage."""
        db_path = self.config.get("sqlite_path", "brainnet_memory.db")
        self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row

        # Create tables
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_session
            ON memories(user_id, session_id)
        """)
        self.sqlite_conn.commit()

    def add(
        self,
        data: dict | str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Add a memory to the store.

        Args:
            data: Memory content (dict or string)
            user_id: Optional user ID override
            session_id: Optional session ID override
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        user_id = user_id or self.user_id
        session_id = session_id or self.session_id

        if isinstance(data, dict):
            content = json.dumps(data)
        else:
            content = str(data)

        if self.use_fallback:
            return self._add_sqlite(content, user_id, session_id, metadata)
        else:
            return self._add_mem0(content, user_id, metadata)

        if self.milvus_manager:
            # Assuming item has an embedding or can generate one
            embedding = data.get('embedding', [])
            if embedding:
                self.milvus_manager.add(
                    embedding=embedding,
                    symbol=data.get('symbol', 'unknown'),
                    decision=data.get('decision', 'flat'),
                    analysis_text=str(data.get('analysis', ''))
                )

    def _add_mem0(self, content: str, user_id: str, metadata: Optional[dict]) -> str:
        """Add memory using Mem0."""
        try:
            result = self.mem0_client.add(
                content,
                user_id=user_id,
                metadata=metadata or {},
            )
            return result.get("id", "")
        except Exception as e:
            print(f"Mem0 add failed: {e}")
            return ""

    def _add_sqlite(
        self,
        content: str,
        user_id: str,
        session_id: str,
        metadata: Optional[dict],
    ) -> str:
        """Add memory using SQLite fallback."""
        cursor = self.sqlite_conn.execute(
            "INSERT INTO memories (user_id, session_id, content, metadata) VALUES (?, ?, ?, ?)",
            (user_id, session_id, content, json.dumps(metadata or {})),
        )
        self.sqlite_conn.commit()
        return str(cursor.lastrowid)

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search memories.

        Args:
            query: Search query
            user_id: Optional user ID filter
            session_id: Optional session ID filter
            limit: Maximum results

        Returns:
            List of matching memories
        """
        user_id = user_id or self.user_id

        if self.use_fallback:
            return self._search_sqlite(query, user_id, session_id, limit)
        else:
            return self._search_mem0(query, user_id, limit)

    def _search_mem0(self, query: str, user_id: str, limit: int) -> list[dict]:
        """Search using Mem0."""
        try:
            results = self.mem0_client.search(
                query,
                user_id=user_id,
                limit=limit,
            )
            return [
                {"content": r.get("memory", r.get("content", "")), "score": r.get("score", 0)}
                for r in results
            ]
        except Exception as e:
            print(f"Mem0 search failed: {e}")
            return []

    def _search_sqlite(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str],
        limit: int,
    ) -> list[dict]:
        """Search using SQLite (simple text matching)."""
        sql = "SELECT content, metadata FROM memories WHERE user_id = ?"
        params = [user_id]

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        # Simple text search
        sql += " AND content LIKE ?"
        params.append(f"%{query}%")

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.sqlite_conn.execute(sql, params)
        results = []
        for row in cursor.fetchall():
            results.append({
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            })
        return results

    def get_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """Get relevant context from memory based on a query.

        Uses Milvus when enabled; otherwise falls back to the existing
        Mem0/SQLite-based search implementation.
        """
        user_id = user_id or self.user_id

        # Milvus-powered retrieval
        if self.milvus_manager:
            query_embedding = self._generate_embedding(query)
            if query_embedding:
                results = self.milvus_manager.get_context(query_embedding, top_k=limit)
                return "\n".join(
                    [
                        f"Past Analysis for {r['symbol']}: Decision - {r['decision']}, Analysis - {r['analysis_text']}"
                        for r in results
                    ]
                )

        # Fallback: use existing vector/text search backend
        matches = self.search(query=query, user_id=user_id, limit=limit)
        if not matches:
            return ""
        return "\n".join(m["content"] for m in matches)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text query using Phi-3.5-vision-instruct via NVIDIA API."""
        import requests
        try:
            invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.config.get('nvidia_api_key', '')}",
                "Accept": "application/json"
            }
            payload = {
                "model": "microsoft/phi-3.5-vision-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Generate a vector embedding for the following text: {text}"
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.20,
                "top_p": 0.70,
                "stream": False
            }
            response = requests.post(invoke_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                # Assuming the API returns an embedding in the response; adjust based on actual API response structure
                embedding = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if isinstance(embedding, list):
                    return embedding
                else:
                    # Parse or convert to list if necessary; placeholder for actual parsing
                    return [float(x) for x in embedding.split(',')] if embedding else [0.0] * self.config.get("embedding_dimension", 1536)
            else:
                return [0.0] * self.config.get("embedding_dimension", 1536)  # Return dummy embedding on failure
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return [0.0] * self.config.get("embedding_dimension", 1536)  # Return dummy embedding on error

    def get_all(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[dict]:
        """Get all memories for a user/session."""
        user_id = user_id or self.user_id

        if self.use_fallback:
            sql = "SELECT content, metadata, created_at FROM memories WHERE user_id = ?"
            params = [user_id]
            if session_id:
                sql += " AND session_id = ?"
                params.append(session_id)
            sql += " ORDER BY created_at DESC"

            cursor = self.sqlite_conn.execute(sql, params)
            return [
                {
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]
        else:
            try:
                return self.mem0_client.get_all(user_id=user_id)
            except Exception:
                return []

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        if self.use_fallback:
            self.sqlite_conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.sqlite_conn.commit()
            return True
        else:
            try:
                self.mem0_client.delete(memory_id)
                return True
            except Exception:
                return False

    def clear(self, user_id: Optional[str] = None) -> bool:
        """Clear all memories for a user."""
        user_id = user_id or self.user_id

        if self.use_fallback:
            self.sqlite_conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            self.sqlite_conn.commit()
            return True
        else:
            try:
                self.mem0_client.delete_all(user_id=user_id)
                return True
            except Exception:
                return False

    def close(self):
        """Close connections."""
        if self.use_fallback and hasattr(self, 'sqlite_conn'):
            self.sqlite_conn.close()
        if self.milvus_manager:
            self.milvus_manager.close()
