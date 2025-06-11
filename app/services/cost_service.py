from app.config import MODEL_PRICES

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates the cost of an LLM request based on model, prompt tokens, and completion tokens."""
    pricing_key = "default"

    best_match_len = 0
    for key in MODEL_PRICES:
        if model_name.startswith(key) and key != "default":
            if len(key) > best_match_len:
                pricing_key = key
                best_match_len = len(key)
    
    if best_match_len == 0 and model_name in MODEL_PRICES: 
        pricing_key = model_name

    prices = MODEL_PRICES.get(pricing_key)

    if not prices: 
        print(f"Warning: Pricing not found for model key '{pricing_key}' (original model: '{model_name}'). Using zero rates.")
        return 0.0
    
    prompt_cost = prompt_tokens * prices["prompt"]
    completion_cost = completion_tokens * prices["completion"]
    total_cost = prompt_cost + completion_cost
    return total_cost 