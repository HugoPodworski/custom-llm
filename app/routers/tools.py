from fastapi import APIRouter, HTTPException
from app.schemas import InventorySearchRequest
from app.tools.inventory import search_inventory_from_vin
from app.config import TECDOC_API_KEY, TECDOC_HOST

router = APIRouter()

@router.post("/search_inventory", tags=["Tools"])
async def search_inventory_by_vin(request: InventorySearchRequest):
    """
    Searches for compatible inventory parts by VIN.

    This endpoint takes a VIN and an optional list of part categories
    (e.g., "Air Filter", "Brake Pads"). It resolves the VIN to a specific
    vehicle, finds compatible parts in those categories via the TecDoc API,
    and then searches the internal Supabase inventory for matching items.

    If `search_categories` is omitted, it will search against all possible
    part categories for the vehicle.
    """
    if not TECDOC_API_KEY or not TECDOC_HOST:
        raise HTTPException(status_code=500, detail="TecDoc API credentials are not configured.")

    try:
        # Directly await the async inventory search function
        result = await search_inventory_from_vin(
            vin_number=request.vin,
            api_key=TECDOC_API_KEY,
            host=TECDOC_HOST,
            search_categories=request.search_categories
        )
        return {
            "vin_searched": request.vin,
            "search_categories": request.search_categories or "all",
            "matches": result
        }
    except RuntimeError as e:
        # Catch specific logic errors from the tool, like VIN not found
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}") 