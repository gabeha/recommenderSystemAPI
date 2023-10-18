def recursive_cast_to_float(obj):
    if isinstance(obj, dict):
        return {k: recursive_cast_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_cast_to_float(v) for v in obj]
    elif isinstance(obj, (int, float)):
        return float(obj)
    else:
        return obj