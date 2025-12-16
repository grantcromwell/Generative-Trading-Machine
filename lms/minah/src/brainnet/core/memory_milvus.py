"""
Milvus integration for Brainnet's MemoryManager to store and retrieve vector embeddings of trading data.
"""

from typing import Optional, List, Dict, Any
from pymilvus import MilvusClient, DataType

from brainnet.core.config import get_config


class MilvusMemoryManager:
    """
    Manages vector storage and retrieval for trading data using Milvus.
    Integrates with Brainnet's MemoryManager for enhanced context retrieval.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.collection_name = self.config.get("milvus_collection", "trading_memory")
        self.dimension = self.config.get("embedding_dimension", 1536)  # Default for many models
        self.client = None
        self._initialize_client()
        self._initialize_collection()

    def _initialize_client(self):
        """Initialize Milvus client based on configuration for local milvus-lite."""
        uri = self.config.get("milvus_uri", "http://localhost:19530")
        token = self.config.get("milvus_token", "")
        try:
            self.client = MilvusClient(uri=uri, token=token)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def _initialize_collection(self):
        """Initialize or load the Milvus collection for trading data."""
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=True)
            schema = self.client.add_field(schema=schema, field_name="id", datatype=DataType.INT64, is_primary=True)
            schema = self.client.add_field(schema=schema, field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            schema = self.client.add_field(schema=schema, field_name="symbol", datatype=DataType.VARCHAR, max_length=50)
            schema = self.client.add_field(schema=schema, field_name="decision", datatype=DataType.VARCHAR, max_length=50)
            schema = self.client.add_field(schema=schema, field_name="analysis_text", datatype=DataType.VARCHAR, max_length=5000)
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="IP",  # Inner Product for similarity
                consistency_level="Bounded"
            )
        self.client.load_collection(self.collection_name)

    def add(self, embedding: List[float], symbol: str, decision: str, analysis_text: str):
        """Add a trading data point with its embedding to Milvus."""
        data = [{
            "embedding": embedding,
            "symbol": symbol,
            "decision": decision,
            "analysis_text": analysis_text
        }]
        self.client.insert(collection_name=self.collection_name, data=data)

    def get_context(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant trading context based on a query embedding."""
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["symbol", "decision", "analysis_text"]
        )
        return [res["entity"] for res in search_results[0]]

    def close(self):
        """Close Milvus client connection."""
        if self.client:
            self.client.close()
