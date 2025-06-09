from fastapi import APIRouter, HTTPException
from app.schemas import VinSearchRequest
from app.tools.inventory import find_vehicle_id_from_vin
from app.config import TECDOC_API_KEY, TECDOC_HOST

router = APIRouter()

@router.post("/search_inventory", tags=["Tools"])
async def search_inventory_by_vin(request: VinSearchRequest):
    """
    Searches for vehicle details by VIN using the TecDoc API.

    This endpoint calls an asynchronous function (`find_vehicle_id_from_vin`)
    that uses async I/O operations.
    """
    if not TECDOC_API_KEY:
        raise HTTPException(status_code=500, detail="TecDoc API key is not configured.")

    try:
        # Directly await the async function
        result = await find_vehicle_id_from_vin(
            vin_number=request.vin,
            api_key=TECDOC_API_KEY,
            host=TECDOC_HOST
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}") 