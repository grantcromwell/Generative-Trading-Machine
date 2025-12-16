"""
Configuration management for Brainnet
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


@dataclass
class ConvNeXtConfig:
    """ConvNeXt model configuration for GAF pattern recognition."""
    
    enabled: bool = False
    model_name: str = "convnext_tiny"
    weights: str = "IMAGENET1K_V1"
    device: str = "auto"  # auto, cuda, mps, cpu
    fine_tuned_path: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "ConvNeXtConfig":
        """Load ConvNeXt configuration from environment variables."""
        load_dotenv()
        return cls(
            enabled=os.getenv("CONVNEXT_ENABLED", "false").lower() == "true",
            model_name=os.getenv("CONVNEXT_MODEL", "convnext_tiny"),
            weights=os.getenv("CONVNEXT_WEIGHTS", "IMAGENET1K_V1"),
            device=os.getenv("CONVNEXT_DEVICE", "auto"),
            fine_tuned_path=os.getenv("CONVNEXT_FINE_TUNED_PATH"),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "weights": self.weights,
            "device": self.device,
            "fine_tuned_path": self.fine_tuned_path,
        }


@dataclass
class PgVectorConfig:
    """PostgreSQL + pgvector configuration for vector memory storage."""
    
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "brainnet"
    collection_name: str = "brainnet_memories"
    embedding_dims: int = 768  # nomic-embed-text default
    
    @classmethod
    def from_env(cls) -> "PgVectorConfig":
        """Load pgvector configuration from environment variables."""
        load_dotenv()
        return cls(
            host=os.getenv("PGVECTOR_HOST", "localhost"),
            port=int(os.getenv("PGVECTOR_PORT", "5432")),
            user=os.getenv("PGVECTOR_USER", "postgres"),
            password=os.getenv("PGVECTOR_PASSWORD", ""),
            database=os.getenv("PGVECTOR_DB", "brainnet"),
            collection_name=os.getenv("PGVECTOR_COLLECTION", "brainnet_memories"),
            embedding_dims=int(os.getenv("PGVECTOR_EMBEDDING_DIMS", "768")),
        )
    
    def get_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        if self.password:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary (masks password)."""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": "***" if self.password else "",
            "database": self.database,
            "collection_name": self.collection_name,
            "embedding_dims": self.embedding_dims,
        }


@dataclass
class BrainnetConfig:
    """Configuration container for Brainnet system."""

    # LLM settings
    llm_backend: str = "local"
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:11434/v1"

    # Memory settings - pgvector is now the default
    memory_db: str = "pgvector"  # pgvector, sqlite
    postgres_url: str = ""
    sqlite_path: str = "brainnet_memory.db"
    
    # pgvector-specific settings
    pgvector_host: str = "localhost"
    pgvector_port: int = 5432
    pgvector_user: str = "postgres"
    pgvector_password: str = ""
    pgvector_database: str = "brainnet"
    pgvector_collection: str = "brainnet_memories"
    embedding_dims: int = 768

    # Milvus configuration for vector storage (using milvus-lite local file)
    milvus_uri: str = "./milvus_lite.db"
    milvus_token: str = ""
    milvus_collection: str = "trading_memory"
    embedding_dimension: int = 1536
    use_milvus: bool = True  # Flag to enable/disable Milvus integration
    nvidia_api_key: str = "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"  # API key for NVIDIA API access for embeddings

    # Coinglass API for derivatives data
    coinglass_api_key: str = "3ebf77531c7045ba8d4868961a554bcc"  # API key for Coinglass derivatives data
    use_coinglass: bool = True  # Enable Coinglass data (funding rate, long/short ratio)

    # Trading settings
    symbol: str = "ES=F"
    confidence_threshold: float = 0.78
    max_refinements: int = 3

    # System settings
    log_level: str = "INFO"
    debug: bool = False

    @classmethod
    def from_env(cls) -> "BrainnetConfig":
        """Load configuration from environment variables."""
        load_dotenv()

        return cls(
            llm_backend=os.getenv("LLM_BACKEND", "local"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            memory_db=os.getenv("MEMORY_DB", "pgvector"),
            postgres_url=os.getenv("POSTGRES_URL", ""),
            sqlite_path=os.getenv("SQLITE_PATH", "brainnet_memory.db"),
            pgvector_host=os.getenv("PGVECTOR_HOST", "localhost"),
            pgvector_port=int(os.getenv("PGVECTOR_PORT", "5432")),
            pgvector_user=os.getenv("PGVECTOR_USER", "postgres"),
            pgvector_password=os.getenv("PGVECTOR_PASSWORD", ""),
            pgvector_database=os.getenv("PGVECTOR_DB", "brainnet"),
            pgvector_collection=os.getenv("PGVECTOR_COLLECTION", "brainnet_memories"),
            embedding_dims=int(os.getenv("PGVECTOR_EMBEDDING_DIMS", "768")),
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            milvus_token=os.getenv("MILVUS_TOKEN", ""),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "trading_memory"),
            embedding_dimension=int(os.getenv("MILVUS_EMBEDDING_DIMENSION", "1536")),
            use_milvus=os.getenv("USE_MILVUS", "false").lower() == "true",
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
            coinglass_api_key=os.getenv("COINGLASS_API_KEY", ""),
            use_coinglass=os.getenv("USE_COINGLASS", "true").lower() == "true",
            symbol=os.getenv("YFINANCE_SYMBOL", "ES=F"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.78")),
            max_refinements=int(os.getenv("MAX_REFINEMENTS", "3")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )

    def get_postgres_url(self) -> str:
        """Build PostgreSQL connection string from config."""
        if self.postgres_url:
            return self.postgres_url
        if self.pgvector_password:
            return f"postgresql://{self.pgvector_user}:{self.pgvector_password}@{self.pgvector_host}:{self.pgvector_port}/{self.pgvector_database}"
        return f"postgresql://{self.pgvector_user}@{self.pgvector_host}:{self.pgvector_port}/{self.pgvector_database}"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "llm_backend": self.llm_backend,
            "llm_api_key": "***" if self.llm_api_key else "",
            "llm_base_url": self.llm_base_url,
            "memory_db": self.memory_db,
            "postgres_url": "***" if self.postgres_url else "",
            "sqlite_path": self.sqlite_path,
            "pgvector_host": self.pgvector_host,
            "pgvector_port": self.pgvector_port,
            "pgvector_user": self.pgvector_user,
            "pgvector_password": "***" if self.pgvector_password else "",
            "pgvector_database": self.pgvector_database,
            "pgvector_collection": self.pgvector_collection,
            "embedding_dims": self.embedding_dims,
            "milvus_uri": self.milvus_uri,
            "milvus_token": "***" if self.milvus_token else "",
            "milvus_collection": self.milvus_collection,
            "embedding_dimension": self.embedding_dimension,
            "use_milvus": self.use_milvus,
            "nvidia_api_key": "***" if self.nvidia_api_key else "",
            "coinglass_api_key": "***" if self.coinglass_api_key else "",
            "use_coinglass": self.use_coinglass,
            "symbol": self.symbol,
            "confidence_threshold": self.confidence_threshold,
            "max_refinements": self.max_refinements,
            "log_level": self.log_level,
            "debug": self.debug,
        }


def get_config() -> dict:
    return BrainnetConfig().to_dict()


def load_config() -> dict:
    """
    Load configuration from environment variables.

    Returns:
        Dictionary with configuration values
    """
    load_dotenv()
    
    # Build postgres_url from components if not explicitly set
    postgres_url = os.getenv("POSTGRES_URL", "")
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

    return {
        "llm_backend": os.getenv("LLM_BACKEND", "local"),
        "llm_api_key": os.getenv("LLM_API_KEY", ""),
        "llm_base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        "memory_db": os.getenv("MEMORY_DB", "pgvector"),
        "postgres_url": postgres_url,
        "sqlite_path": os.getenv("SQLITE_PATH", "brainnet_memory.db"),
        # pgvector settings
        "pgvector_collection": os.getenv("PGVECTOR_COLLECTION", "brainnet_memories"),
        "embedding_dims": int(os.getenv("PGVECTOR_EMBEDDING_DIMS", "768")),
        # Milvus settings
        "milvus_uri": os.getenv("MILVUS_URI", "http://localhost:19530"),
        "milvus_token": os.getenv("MILVUS_TOKEN", ""),
        "milvus_collection": os.getenv("MILVUS_COLLECTION", "trading_memory"),
        "embedding_dimension": int(os.getenv("MILVUS_EMBEDDING_DIMENSION", "1536")),
        "use_milvus": os.getenv("USE_MILVUS", "false").lower() == "true",
        "nvidia_api_key": os.getenv("NVIDIA_API_KEY", ""),
        # Coinglass settings
        "coinglass_api_key": os.getenv("COINGLASS_API_KEY", ""),
        "use_coinglass": os.getenv("USE_COINGLASS", "true").lower() == "true",
        # Trading settings
        "symbol": os.getenv("YFINANCE_SYMBOL", "ES=F"),
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.78")),
        "max_refinements": int(os.getenv("MAX_REFINEMENTS", "3")),
        # ConvNeXt settings
        "convnext_enabled": os.getenv("CONVNEXT_ENABLED", "false").lower() == "true",
        "convnext_model": os.getenv("CONVNEXT_MODEL", "convnext_tiny"),
        "convnext_device": os.getenv("CONVNEXT_DEVICE", "auto"),
        "convnext_weights": os.getenv("CONVNEXT_WEIGHTS", "IMAGENET1K_V1"),
        "convnext_fine_tuned_path": os.getenv("CONVNEXT_FINE_TUNED_PATH"),
    }


def load_convnext_config() -> ConvNeXtConfig:
    """Load ConvNeXt-specific configuration."""
    return ConvNeXtConfig.from_env()


def load_pgvector_config() -> PgVectorConfig:
    """Load pgvector-specific configuration."""
    return PgVectorConfig.from_env()
