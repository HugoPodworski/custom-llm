import httpx
import datetime
import time
import json
from pathlib import Path
from thefuzz import fuzz

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
                    "vehicle_id": None
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
                    "vehicle_id": None
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
                    "vehicle_id": None
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
                    "vehicle_id": None
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
                    "vehicle_id": None
                }

            def find_model():
                """
                Finds the best model match using a two-stage process:
                1. Fuzzy match to find all models with a high similarity score.
                2. Filter the high-score candidates by the target manufacturing year.
                """
                vin_model_name = vin_data['model']
                
                # A high threshold to ensure we only consider very strong matches.
                # This can be tuned if needed.
                MINIMUM_SCORE_THRESHOLD = 85
                
                print(f"\nSearching for models similar to '{vin_model_name}' (Year: {target_year})...")
                
                # --- Stage 1: Find all high-similarity candidates ---
                strong_candidates = []
                for model in model_data['models']:
                    model_name_from_api = model['modelName']
                    
                    # fuzz.token_set_ratio is robust against word order and extra descriptors.
                    score = fuzz.token_set_ratio(vin_model_name, model_name_from_api)
                    
                    if score >= MINIMUM_SCORE_THRESHOLD:
                        strong_candidates.append({'model': model, 'score': score})
                        print(f"  -> Found potential candidate: '{model_name_from_api}' (Score: {score})")

                if not strong_candidates:
                    print("  ✗ No models found with a similarity score above the threshold.")
                    # Fallback to body class matching
                    print("\nTrying body class matching as fallback...")
                    normalized_body_class = target_body_class.upper().replace('-', '').replace('_', '').replace(' ', '')
                    
                    for model in model_data['models']:
                        normalized_api_model = model['modelName'].upper().replace('-', '').replace('_', '').replace(' ', '')
                        if normalized_body_class in normalized_api_model:
                            year_from = int(model['modelYearFrom'][:4])
                            year_to_str = model.get('modelYearTo')
                            year_to = datetime.datetime.now().year if year_to_str is None else int(year_to_str[:4])
                            
                            if year_from <= target_year <= year_to:
                                print(f"  ✓ Found body class match: {model['modelName']} (ID: {model['modelId']})")
                                return model['modelId']
                    return None

                # --- Stage 2: Filter candidates by year and find the best one ---
                print(f"\nFound {len(strong_candidates)} strong candidate(s). Filtering by year ({target_year})...")
                
                best_match = None
                highest_score_in_year = 0

                for candidate in strong_candidates:
                    model_details = candidate['model']
                    
                    year_from = int(model_details['modelYearFrom'][:4])
                    year_to_str = model_details.get('modelYearTo')
                    year_to = datetime.datetime.now().year if year_to_str is None else int(year_to_str[:4])

                    # Check if the target year is within the model's production range
                    if year_from <= target_year <= year_to:
                        print(f"  ✓ Candidate '{model_details['modelName']}' has a matching year range ({year_from}-{year_to}).")
                        # If this is the best-scoring model within the correct year, select it.
                        if candidate['score'] > highest_score_in_year:
                            highest_score_in_year = candidate['score']
                            best_match = model_details
                    else:
                        print(f"  ✗ Candidate '{model_details['modelName']}' has an incorrect year range ({year_from}-{year_to}).")

                if best_match:
                    print(f"\n✓ Selected final model: '{best_match['modelName']}' (ID: {best_match['modelId']})")
                    return best_match['modelId']
                else:
                    print("\n✗ Found strong name matches, but none were in the correct year range.")
                    return None

            potential_model_id = time_processing_step("Find Model", find_model)
            if not potential_model_id:
                final_result_message = "Could not find a matching model for the specified year and model/body type."
                return {
                    "vin_searched": vin_number,
                    "result": final_result_message,
                    "total_duration_seconds": round(time.time() - start_time, 3),
                    "timing_details": timings,
                    "vehicle_id": None
                }

            print(f"\nStep 3: Getting all vehicle types for Model ID {potential_model_id}...")
            types_endpoint = f"types/list-vehicles-types/{potential_model_id}/manufacturer-id/{manufacturer_id}/lang-id/4/country-filter-id/{country_filter_id}/type-id/1"
            types_data = await time_api_call(client, types_endpoint)
            if not types_data or 'modelTypes' not in types_data:
                final_result_message = "Failed at Vehicle Types step or no variants found."
                return {
                    "vin_searched": vin_number,
                    "result": final_result_message,
                    "total_duration_seconds": round(time.time() - start_time, 3),
                    "timing_details": timings,
                    "vehicle_id": None
                }

            print(f"\nFound {types_data['countModelTypes']} variants. Filtering...")

            def find_exact_variant():
                for vehicle in types_data['modelTypes']:
                    print(f"\nChecking variant: {vehicle['typeEngineName']}")
                    
                    # Check year range
                    year_from = int(vehicle['constructionIntervalStart'][:4])
                    year_to_str = vehicle['constructionIntervalEnd']
                    year_to = datetime.datetime.now().year if year_to_str is None else int(year_to_str[:4])
                    if not (year_from <= target_year <= year_to):
                        print(f"  ✗ Year out of range ({year_from}-{year_to})")
                        continue
                    print(f"  ✓ Year in range: {year_from}-{year_to}")

                    # Check cylinders
                    if int(float(vehicle.get('numberOfCylinders', 0))) != target_cylinders:
                        print(f"  ✗ Cylinder count mismatch: {vehicle.get('numberOfCylinders')} vs {target_cylinders}")
                        continue
                    print(f"  ✓ Cylinder count matches: {target_cylinders}")

                    # Check displacement with a 2% tolerance to handle marketing vs technical values
                    capacity_lt_from_api = float(vehicle.get('capacityLt', 0.0))
                    DISP_TOLERANCE = 0.02  # 2% tolerance
                    disp_lower_bound = target_displacement * (1 - DISP_TOLERANCE)
                    disp_upper_bound = target_displacement * (1 + DISP_TOLERANCE)

                    if not (disp_lower_bound <= capacity_lt_from_api <= disp_upper_bound):
                        print(f"  ✗ Displacement mismatch: API has {capacity_lt_from_api}L, VIN target is {target_displacement}L")
                        continue
                    print(f"  ✓ Displacement matches: API has {capacity_lt_from_api}L, within tolerance of VIN's {target_displacement}L")

                    # Check engine code (skip if VIN didn't provide engine code)
                    if target_engine_code and target_engine_code not in vehicle.get('engineCodes', ''):
                        print(f"  ✗ Engine code mismatch: {vehicle.get('engineCodes')} vs {target_engine_code}")
                        continue
                    if target_engine_code:
                        print(f"  ✓ Engine code matches: {target_engine_code}")
                    else:
                        print(f"  ⚠ Engine code not available from VIN, skipping engine code check")
                    
                    # Check power (with 5% tolerance)
                    power_ps = float(vehicle.get('powerPs', 0.0))
                    if not (target_power_hp * 0.98 <= power_ps <= target_power_hp * 1.05):
                        print(f"  ✗ Power mismatch: {power_ps}PS vs {target_power_hp}HP")
                        continue
                    print(f"  ✓ Power matches: {power_ps}PS ≈ {target_power_hp}HP")

                    # Check drive type
                    is_4wd = "4WD" in vehicle.get('typeEngineName', '').upper()
                    if target_drive_type == '4x2' and is_4wd:
                        print(f"  ✗ Drive type mismatch: 4WD vs 4x2")
                        continue
                    if target_drive_type == '4x4' and not is_4wd:
                        print(f"  ✗ Drive type mismatch: 2WD vs 4x4")
                        continue
                    print(f"  ✓ Drive type matches: {target_drive_type}")

                    print(f"\nSUCCESS: Found exact match!")
                    print(f"Vehicle Name: {vehicle['typeEngineName']}")
                    return vehicle['vehicleId']
                
                return None

            final_vehicle_id = time_processing_step("Find Exact Variant", find_exact_variant)
            
            if final_vehicle_id:
                final_result_message = f"Successfully found vehicle_id: '{final_vehicle_id}'"
            else:
                final_result_message = "No exact vehicle match found after filtering."

        except Exception as e:
            final_result_message = f"An unexpected error occurred during the search: {str(e)}"

    total_duration = time.time() - start_time
    return {
        "vin_searched": vin_number,
        "result": final_result_message,
        "total_duration_seconds": round(total_duration, 3),
        "timing_details": timings,
        "vehicle_id": final_vehicle_id
    }