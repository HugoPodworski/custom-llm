from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Pydantic models for /search endpoint
class SearchRequest(BaseModel):
    query_text: str
    top_k: Optional[int] = 5

class ScenarioHit(BaseModel):
    id: Any
    score: float
    payload: Dict[str, Any]

# Pydantic model for VIN search endpoint
class VinSearchRequest(BaseModel):
    vin: str = Field(..., description="The 17-character Vehicle Identification Number.", min_length=17, max_length=17)

# Pydantic model for inventory search via VIN
class InventorySearchRequest(BaseModel):
    vin: str = Field(..., description="The 17-character Vehicle Identification Number.", min_length=17, max_length=17)
    search_categories: Optional[List[str]] = Field(None, description="A list of part categories to search for, e.g., 'Air Filter'. If omitted, all categories will be searched.") 