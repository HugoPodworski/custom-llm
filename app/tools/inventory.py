import httpx
import datetime
import time
import json
import re
from pathlib import Path
from thefuzz import fuzz
import os
import asyncio
from supabase import create_client, Client

# IMPORTANT: This entire file now uses asynchronous, non-blocking I/O (httpx).
# It can be called directly from FastAPI async endpoints.

async def find_vehicle_id_from_vin(vin_number: str, api_key: str, host: str) -> dict:
    """
    Takes a VIN, decodes it, and steps through the TecDoc API to find the vehicleId.
    This function uses fuzzy string matching and improved tolerance handling for better accuracy.
    """
    # --- Configuration ---
    HEADERS = {
        "x-rapidapi-host": host,
        "x-rapidapi-key": api_key
    }
    CACHE_DIR = Path("cache")
    COUNTRIES_CACHE_FILE = CACHE_DIR / "countries.json"
    CACHE_EXPIRY_DAYS = 30
    CACHE_DIR.mkdir(exist_ok=True)
    timings = {'api_calls': {}, 'processing_steps': {}}

    # --- Helper Functions ---
    async def call_tecdoc_api(client, endpoint_path):
        """A helper function to handle calls to the TecDoc API with improved error logging."""
        url = f"https://{host}/{endpoint_path}"
        try:
            print(f"  -> Calling API: {url}")
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            result = response.json()
            print(f"Response: {result}")
            return result
        except httpx.HTTPStatusError as http_err:
            print(f"HTTP Error occurred: {http_err}")
            print(f"Status Code: {http_err.response.status_code}")
            try:
                print(f"Error Body: {http_err.response.json()}")
            except ValueError:
                print(f"Error Body: {http_err.response.text}")
            return None
        except httpx.RequestError as req_err:
            print(f"Request Error occurred: {req_err}")
            return None
        except Exception as e:
            print(f"API Call Error occurred: {e}")
            return None

    async def time_api_call(client, endpoint_path):
        """Time an API call and store the result"""
        start = time.time()
        result = await call_tecdoc_api(client, endpoint_path)
        end = time.time()
        timings['api_calls'][endpoint_path] = end - start
        return result

    def time_processing_step(step_name, func, *args, **kwargs):
        """Time a processing step and store the result"""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        timings['processing_steps'][step_name] = end - start
        return result

    async def get_cached_countries(client):
        """Get countries list from cache or API"""
        # Check if cache exists and is fresh
        if COUNTRIES_CACHE_FILE.exists():
            cache_age = time.time() - COUNTRIES_CACHE_FILE.stat().st_mtime
            if cache_age < CACHE_EXPIRY_DAYS * 24 * 60 * 60:
                with open(COUNTRIES_CACHE_FILE, 'r') as f:
                    return json.load(f)
        
        # If cache doesn't exist or is stale, fetch from API
        countries_response = await time_api_call(client, "countries/list")
        if countries_response:
            with open(COUNTRIES_CACHE_FILE, 'w') as f:
                json.dump(countries_response, f)
        return countries_response

    def get_country_filter_id(plant_country, countries_response):
        """Get the TecDoc country filter ID based on the vehicle's manufacturing country."""
        print(f"\nLooking up country filter ID for: {plant_country}")
        
        if not countries_response or 'countries' not in countries_response:
            print("  ⚠ Failed to get countries list, defaulting to USA (223)")
            return 223

        # Some common country name variations
        country_variations = {
            "UNITED STATES": ["USA", "UNITED STATES OF AMERICA", "U.S.A.", "US"],
            "JAPAN": ["JPN", "NIPPON"],
            "UNITED KINGDOM": ["UK", "GREAT BRITAIN", "GB"],
            # Add more variations as needed
        }

        # Normalize the plant country
        plant_country_upper = plant_country.upper()
        
        for country in countries_response['countries']:
            country_name = country['countryName'].upper()
            
            # Direct match
            if country_name == plant_country_upper:
                print(f"  ✓ Found exact country match: {country['countryName']} (ID: {country['id']})")
                return country['id']
            
            # Check variations
            for standard_name, variations in country_variations.items():
                if plant_country_upper in variations or standard_name.upper() == plant_country_upper:
                    if standard_name.upper() in country_name:
                        print(f"  ✓ Found country match through variation: {country['countryName']} (ID: {country['id']})")
                        return country['id']
        
        print("  ⚠ Could not find matching country, defaulting to USA (223)")
        return 223  # Default to USA if no match found

    # --- Main Logic ---
    start_time = time.time()
    final_result_message = "Search could not be completed."
    final_vehicle_id = None
    country_filter_id = None  # we expose this later so callers can reuse it

    async with httpx.AsyncClient(headers=HEADERS) as client:
        try:
            print(f"Starting search for VIN: {vin_number}")

            # --- STEP 0: VIN DECODING ---
            print("\nStep 0: Decoding VIN...")
            vin_endpoint = f"vin/decoder-v2/{vin_number}"
            decoded_response = await time_api_call(client, vin_endpoint)

            if not decoded_response:
                final_result_message = f"VIN decoding failed for {vin_number}. API call failed."
                return {
                    "vin_searched": vin_number,
                    "result": final_result_message,
                    "total_duration_seconds": round(time.time() - start_time, 3),
                    "timing_details": timings,
                    "vehicle_id": None,
                    "country_filter_id": country_filter_id,
                }
            
            # Check if critical fields are present
            critical_fields = ['make', 'model_year', 'body_class', 'plant_country']
            missing_fields = [field for field in critical_fields if not decoded_response.get(field)]
            
            if missing_fields:
                final_result_message = f"VIN decoding incomplete for {vin_number}. Missing critical fields: {', '.join(missing_fields)}"
                return {
                    "vin_searched": vin_number,
                    "result": final_result_message,
                    "total_duration_seconds": round(time.time() - start_time, 3),
                    "timing_details": timings,
                    "vehicle_id": None,
                    "country_filter_id": country_filter_id,
                }
            
            vin_data = decoded_response
            print("VIN Decoded Successfully.")

            # Get countries from cache
            def get_countries_wrapper():
                return None  # Will be set by async call
            countries_response = await get_cached_countries(client)

            # Get country filter ID
            def get_country_wrapper():
                return get_country_filter_id(vin_data['plant_country'], countries_response)
            country_filter_id = time_processing_step("Get Country Filter", get_country_wrapper)

            def extract_vehicle_data():
                nonlocal vin_data
                target_make = vin_data.get("make")
                target_year = int(vin_data.get("model_year", 0))
                target_body_class = vin_data.get("body_class")
                target_cylinders = int(float(vin_data.get("engine_number_of_cylinders", 0)))
                target_displacement = float(vin_data.get("displacement_(l)", 0.0))
                target_engine_code = vin_data.get("engine_model")
                target_power_hp = int(float(vin_data.get("engine_brake_(hp)_from", 0)))
                target_drive_type = vin_data.get("drive_type")
                return (target_make, target_year, target_body_class, target_cylinders,
                       target_displacement, target_engine_code, target_power_hp, target_drive_type)

            vehicle_data = time_processing_step("Extract Vehicle Data", extract_vehicle_data)
            (target_make, target_year, target_body_class, target_cylinders,
             target_displacement, target_engine_code, target_power_hp, target_drive_type) = vehicle_data

            print(f"\nVIN Fingerprint Created:")
            print(f"  Make: {target_make}")
            print(f"  Year: {target_year}")
            print(f"  Body Class: {target_body_class}")
            print(f"  Engine: {target_displacement}L {target_cylinders}-Cyl")
            print(f"  Engine Code: {target_engine_code}")
            print(f"  Power: {target_power_hp} HP")
            print(f"  Drive Type: {target_drive_type}")
            print(f"  Manufacturing Country: {vin_data['plant_country']} (Filter ID: {country_filter_id})")

            print("\nStep 1: Getting Manufacturer ID...")
            mfr_endpoint = f"manufacturers/list/lang-id/4/country-filter-id/{country_filter_id}/type-id/1"
            mfr_data = await time_api_call(client, mfr_endpoint)
            if not mfr_data:
                final_result_message = "Failed at Manufacturer step."
                return {
                    "vin_searched": vin_number,
                    "result": final_result_message,
                    "total_duration_seconds": round(time.time() - start_time, 3),
                    "timing_details": timings,
                    "vehicle_id": None,
                    "country_filter_id": country_filter_id,
                }

            def find_manufacturer():
                for manufacturer in mfr_data['manufacturers']:
                    if manufacturer['brand'].upper() == target_make.upper():
                        print(f"  ✓ Found match: {manufacturer['brand']} (ID: {manufacturer['manufacturerId']})")
                        return manufacturer['manufacturerId']
                return None

            manufacturer_id = time_processing_step("Find Manufacturer", find_manufacturer)
            if not manufacturer_id:
                final_result_message = f"Could not find Manufacturer ID for {target_make}"
                return {
                    "vin_searched": vin_number,
                    "result": final_result_message,
                    "total_duration_seconds": round(time.time() - start_time, 3),
                    "timing_details": timings,
                    "vehicle_id": None,
                    "country_filter_id": country_filter_id,
                }

            print("\nStep 2: Getting Model ID...")
            model_endpoint = f"models/list/manufacturer-id/{manufacturer_id}/lang-id/4/country-filter-id/{country_filter_id}/type-id/1"
            model_data = await time_api_call(client, model_endpoint)
            if not model_data:
                final_result_message = "Failed at Model step."
                return {
                    "vin_searched": vin_number,
                    "result": final_result_message,
                    "total_duration_seconds": round(time.time() - start_time, 3),
                    "timing_details": timings,
                    "vehicle_id": None,
                    "country_filter_id": country_filter_id,
                }

            def find_model_candidates():
                """
                Finds the best model match candidates using a multi-stage process.
                1. Fuzzy match to find all models with a high similarity score.
                2. Filter the high-score candidates by the target manufacturing year.
                3. Sort the valid candidates by score (desc) and year range specificity (asc).
                Returns a list of candidate model dicts.
                """
                vin_model_name = vin_data['model']
                MINIMUM_SCORE_THRESHOLD = 85
                print(f"\nSearching for models similar to '{vin_model_name}' (Year: {target_year})...")

                # --- Stage 1: Find all high-similarity candidates ---
                strong_candidates = []
                for model in model_data['models']:
                    model_name_from_api = model['modelName']
                    score = fuzz.token_set_ratio(vin_model_name, model_name_from_api)
                    if score >= MINIMUM_SCORE_THRESHOLD:
                        strong_candidates.append({'model': model, 'score': score})
                        print(f"  -> Found potential candidate: '{model_name_from_api}' (Score: {score})")

                if not strong_candidates:
                    print("  ✗ No models found with a similarity score above the threshold.")
                    return None

                # --- Stage 2: Filter candidates by year ---
                print(f"\nFound {len(strong_candidates)} strong candidate(s). Filtering by year ({target_year})...")
                valid_candidates = []
                for candidate in strong_candidates:
                    model_details = candidate['model']
                    year_from = int(model_details['modelYearFrom'][:4])
                    year_to_str = model_details.get('modelYearTo')
                    year_to = datetime.datetime.now().year if year_to_str is None else int(year_to_str[:4])

                    if year_from <= target_year <= year_to:
                        print(f"  ✓ Candidate '{model_details['modelName']}' has a matching year range ({year_from}-{year_to}).")
                        range_size = year_to - year_from
                        valid_candidates.append({
                            'model': model_details,
                            'score': candidate['score'],
                            'year_range_size': range_size
                        })
                    else:
                        print(f"  ✗ Candidate '{model_details['modelName']}' has an incorrect year range ({year_from}-{year_to}).")

                if not valid_candidates:
                    print("\n✗ Found strong name matches, but none were in the correct year range.")
                    return None

                # --- Stage 3: Sort by score and year range specificity ---
                valid_candidates.sort(key=lambda x: (-x['score'], x['year_range_size']))
                print(f"\nSorted valid candidates:")
                for i, c in enumerate(valid_candidates):
                    print(f"  {i+1}. '{c['model']['modelName']}' (Score: {c['score']}, Range: {c['year_range_size']} years)")
                
                return [c['model'] for c in valid_candidates]

            potential_model_candidates = time_processing_step("Find Model Candidates", find_model_candidates)
            if not potential_model_candidates:
                final_result_message = "Could not find a matching model for the specified year and model/body type."
                return {
                    "vin_searched": vin_number,
                    "result": final_result_message,
                    "total_duration_seconds": round(time.time() - start_time, 3),
                    "timing_details": timings,
                    "vehicle_id": None,
                    "country_filter_id": country_filter_id,
                }

            # --- MODIFICATION: Iterate through model candidates ---
            final_vehicle_id = None
            for i, model_candidate in enumerate(potential_model_candidates):
                model_id = model_candidate['modelId']
                model_name = model_candidate['modelName']
                print(f"\n{'='*60}")
                print(f"Trying model candidate {i+1}/{len(potential_model_candidates)}: '{model_name}' (ID: {model_id})")
                print(f"{'='*60}")
                
                types_endpoint = f"types/list-vehicles-types/{model_id}/manufacturer-id/{manufacturer_id}/lang-id/4/country-filter-id/{country_filter_id}/type-id/1"
                types_data = await time_api_call(client, types_endpoint)

                if not types_data or 'modelTypes' not in types_data:
                    print(f"  ✗ Failed to get vehicle types for model '{model_name}', trying next candidate...")
                    continue
                
                print(f"\nFound {types_data['countModelTypes']} variants. Filtering...")

                def find_exact_variant():
                    for vehicle in types_data['modelTypes']:
                        # This inner logic remains largely the same, checking year, cylinders, etc.
                        print(f"\nChecking variant: {vehicle['typeEngineName']}")
                        
                        year_from = int(vehicle['constructionIntervalStart'][:4])
                        year_to_str = vehicle.get('constructionIntervalEnd')
                        year_to = datetime.datetime.now().year if year_to_str is None else int(year_to_str[:4])
                        if not (year_from <= target_year <= year_to):
                            print(f"  ✗ Year out of range ({year_from}-{year_to})")
                            continue
                        print(f"  ✓ Year in range: {year_from}-{year_to}")

                        if int(float(vehicle.get('numberOfCylinders', 0))) != target_cylinders:
                            print(f"  ✗ Cylinder count mismatch: {vehicle.get('numberOfCylinders')} vs {target_cylinders}")
                            continue
                        print(f"  ✓ Cylinder count matches: {target_cylinders}")

                        capacity_lt_from_api = float(vehicle.get('capacityLt', 0.0))
                        DISP_TOLERANCE = 0.02
                        disp_lower_bound = target_displacement * (1 - DISP_TOLERANCE)
                        disp_upper_bound = target_displacement * (1 + DISP_TOLERANCE)
                        if not (disp_lower_bound <= capacity_lt_from_api <= disp_upper_bound):
                            print(f"  ✗ Displacement mismatch: API has {capacity_lt_from_api}L, VIN target is {target_displacement}L")
                            continue
                        print(f"  ✓ Displacement matches: API has {capacity_lt_from_api}L, within tolerance of VIN's {target_displacement}L")

                        if target_engine_code and target_engine_code not in vehicle.get('engineCodes', ''):
                            print(f"  ✗ Engine code mismatch: {vehicle.get('engineCodes')} vs {target_engine_code}")
                            continue
                        if target_engine_code:
                            print(f"  ✓ Engine code matches: {target_engine_code}")
                        else:
                            print(f"  ⚠ Engine code not available from VIN, skipping engine code check")
                        
                        power_ps = float(vehicle.get('powerPs', 0.0))
                        if target_power_hp > 0 and not (target_power_hp * 0.95 <= power_ps <= target_power_hp * 1.05):
                            print(f"  ✗ Power mismatch: {power_ps}PS vs {target_power_hp}HP")
                            continue
                        if target_power_hp > 0:
                            print(f"  ✓ Power matches: {power_ps}PS ≈ {target_power_hp}HP")
                        else:
                             print(f"  ⚠ Power not available from VIN, skipping power check")

                        type_engine_name = vehicle.get('typeEngineName', '').upper()
                        is_4wd = any(term in type_engine_name for term in ['4WD', 'AWD', '4X4'])
                        if target_drive_type == '4x2' and is_4wd:
                            print(f"  ✗ Drive type mismatch: Found 4WD/AWD in '{vehicle.get('typeEngineName')}' vs VIN indicates 4x2")
                            continue
                        if target_drive_type == '4x4' and not is_4wd:
                            print(f"  ✗ Drive type mismatch: No 4WD/AWD found in '{vehicle.get('typeEngineName')}' vs VIN indicates 4x4")
                            continue
                        print(f"  ✓ Drive type compatible: {target_drive_type}")

                        print(f"\n🎉 SUCCESS: Found exact match!")
                        print(f"Vehicle Name: {vehicle['typeEngineName']}")
                        print(f"Model: {model_name}")
                        print(f"Vehicle ID: {vehicle['vehicleId']}")
                        return vehicle['vehicleId']
                    
                    return None

                final_vehicle_id = time_processing_step("Find Exact Variant", find_exact_variant)
                if final_vehicle_id:
                    break # Found a match, stop trying other models
                else:
                    print(f"\n  ✗ No exact variants found for model '{model_name}', trying next candidate...")
            
            if final_vehicle_id:
                final_result_message = f"Successfully found vehicle_id: '{final_vehicle_id}'"
            else:
                final_result_message = "No exact vehicle match found after trying all model candidates."

        except Exception as e:
            final_result_message = f"An unexpected error occurred during the search: {str(e)}"

    total_duration = time.time() - start_time
    return {
        "vin_searched": vin_number,
        "result": final_result_message,
        "total_duration_seconds": round(total_duration, 3),
        "timing_details": timings,
        "vehicle_id": final_vehicle_id,
        "country_filter_id": country_filter_id,
    }

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Initialise the Supabase client once so we can re-use the HTTP keep-alive pool
_supabase: Client | None = None

def _get_supabase_client():
    """Return a singleton Supabase client (the library is synchronous)."""
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY environment variables not set")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase

def _normalize_article_no(s: str) -> str:
    """Converts to uppercase and removes all non-alphanumeric characters."""
    if not s:
        return ""
    return re.sub(r'[^A-Z0-9]', '', s.upper())

async def search_inventory_from_vin(
    vin_number: str,
    api_key: str,
    host: str,
    *,
    search_categories: list[str] | None = None,
) -> list[dict]:
    """High-level helper that ties VIN->vehicle->TecDoc categories together and
    finally queries Supabase inventory for compatible parts.

    Returns a list with one entry per matching inventory item. Each entry is a
    dict with:
        inventory_item – the row from Supabase
        compatible_article – the matching TecDoc article
        match_type – 'exact' | 'fuzzy'
        similarity – fuzz score (0-100)
    """

    # Step 1 – find the TecDoc vehicle id & country filter id
    vehicle_info = await find_vehicle_id_from_vin(vin_number, api_key, host)
    vehicle_id = vehicle_info.get("vehicle_id")
    country_filter_id = vehicle_info.get("country_filter_id") or 223  # default USA
    if not vehicle_id:
        raise RuntimeError(f"Could not resolve VIN {vin_number} to a vehicle id: {vehicle_info.get('result')}")

    # Convenience: if caller did not specify categories, fetch all available ones
    # from TecDoc and use every leaf category.
    async with httpx.AsyncClient(headers={
        "x-rapidapi-host": host,
        "x-rapidapi-key": api_key,
    }) as client:
        async def call_tecdoc(path: str):
            url = f"https://{host}/{path}"
            r = await client.get(url, timeout=10)
            r.raise_for_status()
            return r.json()

        # Fetch category tree
        categories_endpoint = (
            f"category/category-products-groups-variant-3/{vehicle_id}/manufacturer-id/5/"
            f"lang-id/4/country-filter-id/{country_filter_id}/type-id/1"
        )
        categories_data = await call_tecdoc(categories_endpoint)
        if not categories_data or "categories" not in categories_data:
            raise RuntimeError("TecDoc categories lookup failed")

        # Helper to flatten nested category dict {id: {text, children}}
        def _collect_cat_ids(cat_dict: dict, acc: set[str]):
            for cid, info in cat_dict.items():
                acc.add(cid)
                if info.get("children"):
                    _collect_cat_ids(info["children"], acc)

        all_category_ids: set[str] = set()
        _collect_cat_ids(categories_data["categories"], all_category_ids)

        # If user passed an explicit subset, we need to map the human names -> ids
        if search_categories:
            desired_ids: set[str] = set()
            
            # --- MODIFICATION: Recursive category search ---
            def find_best_category_id(search_name: str) -> str | None:
                """Recursively search the category tree for the best fuzzy match."""
                best_match_info = {'id': None, 'score': 0}

                def search_recursive(categories: dict):
                    for cat_id, cat_info in categories.items():
                        cat_name = cat_info.get("text", "")
                        score = fuzz.token_set_ratio(search_name, cat_name)
                        if score > best_match_info['score']:
                            best_match_info['score'] = score
                            best_match_info['id'] = cat_id
                        
                        if "children" in cat_info and cat_info["children"]:
                            search_recursive(cat_info["children"])
                
                search_recursive(categories_data["categories"])
                
                MIN_CAT_SCORE = 70
                if best_match_info['score'] >= MIN_CAT_SCORE:
                    return best_match_info['id']
                return None

            for human_name in search_categories:
                print(f"--> Searching for category ID for '{human_name}'...")
                best_id = find_best_category_id(human_name)
                if best_id:
                    print(f"<-- Found best match category ID: {best_id}")
                    desired_ids.add(best_id)
                else:
                    print(f"<-- No suitable category found for '{human_name}'")

            category_ids = desired_ids
        else:
            category_ids = all_category_ids

        supabase = _get_supabase_client()
        matches: list[dict] = []

        # Fetch the entire inventory from Supabase ONCE at the start.
        def _fetch_inventory():
            print("--> Fetching all items from Supabase inventory...")
            response = supabase.table("inventory").select("*").execute()
            return response.data or []

        inventory_items = await asyncio.to_thread(_fetch_inventory)
        print(f"<-- Found {len(inventory_items)} items in Supabase.")
        if inventory_items:
            # Log the first item to help debug article number formats
            print(f"    Example inventory item: {inventory_items[0]}")

        print(f"--> Searching TecDoc for {len(category_ids)} category IDs: {list(category_ids)}")
        for category_id in category_ids:
            articles_endpoint = (
                f"articles/list/vehicle-id/{vehicle_id}/product-group-id/{category_id}/"
                f"manufacturer-id/5/lang-id/4/country-filter-id/{country_filter_id}/type-id/1"
            )
            try:
                print(f"--> Getting articles for category_id: {category_id}")
                articles_data = await call_tecdoc(articles_endpoint)
            except httpx.HTTPStatusError:
                print(f"<-- Skipping category {category_id} due to HTTP error.")
                continue

            if not articles_data or "articles" not in articles_data or articles_data.get("articles") is None:
                print(f"<-- No articles found for category_id: {category_id}")
                continue

            print(f"<-- Found {len(articles_data['articles'])} compatible articles for category {category_id}.")

            # Build a quick lookup of compatible article numbers, using a normalized version as the key.
            compat = {_normalize_article_no(a["articleNo"]): a for a in articles_data["articles"]}
            compat_numbers = set(compat.keys())
            if compat_numbers:
                print(f"    Compatible TecDoc article numbers (normalized sample): {list(compat_numbers)[:5]}")

            # No longer fetching inventory inside the loop. We use the list fetched earlier.
            for item in inventory_items:
                # Normalize the inventory article number for comparison
                inv_no_raw = item.get("article_no", "")
                inv_no_normalized = _normalize_article_no(inv_no_raw)
                
                if not inv_no_normalized:
                    continue
                
                # Log the comparison being made
                # print(f"    Comparing Supabase item '{inv_no_raw}' (normalized: '{inv_no_normalized}') with TecDoc parts...")

                if inv_no_normalized in compat_numbers:
                    print(f"    ✓✓✓ Exact match found: Inventory='{inv_no_raw}' (normalized: {inv_no_normalized})")
                    matches.append({
                        "inventory_item": item,
                        "compatible_article": compat[inv_no_normalized],
                        "match_type": "exact",
                        "similarity": 100,
                    })
                else:
                    # fuzzy match on normalized strings
                    best_article, best_score = None, 0
                    for art_no_normalized, art in compat.items():
                        score = fuzz.ratio(inv_no_normalized, art_no_normalized)
                        if score > best_score:
                            best_article, best_score = art, score
                    if best_score >= 85:
                        print(f"    ✓✓✓ Fuzzy match found: Inventory='{inv_no_raw}' --- TecDoc='{best_article['articleNo']}' (Score: {best_score})")
                        matches.append({
                            "inventory_item": item,
                            "compatible_article": best_article,
                            "match_type": "fuzzy",
                            "similarity": best_score,
                        })

        return matches