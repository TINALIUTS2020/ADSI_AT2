def safe_convert_float(var, target_type=float, default=-9):
    try:
        return target_type(var)
    except (ValueError, TypeError) as e:
        return target_type(default)
