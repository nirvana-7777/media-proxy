from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from enum import Enum


class DecryptionAlgorithm(str, Enum):
    AES_128_CTR = "aes-128-ctr"
    AES_128_CBC = "aes-128-cbc"
    AES_256_CBC = "aes-256-cbc"


class DecryptRequest(BaseModel):
    key: str = Field(..., description="Hex-encoded decryption key (32 chars for AES-128, 64 chars for AES-256)")
    url: HttpUrl = Field(..., description="URL of the MP4 segment to decrypt")
    iv: Optional[str] = Field(None, description="Base64 encoded initialization vector")
    algorithm: DecryptionAlgorithm = Field(
        default=DecryptionAlgorithm.AES_128_CTR,
        description="Decryption algorithm to use"
    )
    remove_protection_boxes: bool = Field(
        default=True,
        description="Remove encryption metadata boxes (senc, tenc, pssh, etc.)"
    )


class DecryptResponse(BaseModel):
    success: bool
    data_size: Optional[int] = Field(None, description="Size of decrypted data in bytes")
    error: Optional[str] = None
    processing_time: float
    samples_processed: Optional[int] = Field(None, description="Number of samples processed")
    kid: Optional[str] = Field(None, description="Key ID found in the segment")
    pssh_boxes: Optional[List[str]] = Field(None, description="PSSH boxes found in the segment")


class BatchDecryptRequest(BaseModel):
    requests: List[DecryptRequest] = Field(..., max_items=100)


class BatchDecryptResponse(BaseModel):
    results: List[DecryptResponse]
    total_processed: int
    total_succeeded: int
    total_failed: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    memory_usage: float
    active_tasks: int


class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[DecryptResponse] = None


class DecryptorInfo(BaseModel):
    algorithm: DecryptionAlgorithm
    supports_ctr: bool = True
    supports_cbc: bool = True
    max_concurrent_downloads: int
    cache_enabled: bool
    cache_size: int