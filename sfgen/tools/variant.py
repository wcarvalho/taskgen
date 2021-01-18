def update_config(default, variant, strict=False):
    """Performs deep update on all dict structures from ``variant``, updating only
    individual fields.  Any field in ``variant`` must be present in ``default``,
    else raises ``KeyError`` (helps prevent mistakes).  Operates recursively to
    return a new dictionary."""
    new = default.copy()
    for k, v in variant.items():
        if strict:
            if k not in new:
                raise KeyError(f"Variant key {k} not found in default config.")
            if isinstance(v, dict) != isinstance(new[k], dict):
                raise TypeError(f"Variant dict structure at key {k} mismatched with"
                    " default.")
        new[k] = update_config(new[k], v, strict=strict) if isinstance(v, dict) else v
    return new