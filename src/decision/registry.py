"""Small explicit registry for opt-in decision policies."""


_POLICY_CLASSES = {}
_ALIASES = {}


def register_policy(name, policy_class, *, aliases=()):
    """Register a policy class under one canonical name and optional aliases."""

    canonical = str(name).strip().lower()
    if not canonical:
        raise ValueError("Policy name must not be empty.")
    existing = _POLICY_CLASSES.get(canonical)
    if existing is not None and existing is not policy_class:
        raise ValueError(f"Decision policy '{canonical}' is already registered.")
    _POLICY_CLASSES[canonical] = policy_class
    for alias in aliases:
        normalized = str(alias).strip().lower()
        if not normalized:
            raise ValueError("Policy alias must not be empty.")
        current = _ALIASES.get(normalized)
        if current is not None and current != canonical:
            raise ValueError(f"Decision policy alias '{normalized}' is in use.")
        _ALIASES[normalized] = canonical
    return policy_class


def canonical_policy_name(name):
    """Resolve a canonical registry key or raise a helpful error."""

    normalized = str(name).strip().lower()
    canonical = _ALIASES.get(normalized, normalized)
    if canonical not in _POLICY_CLASSES:
        available = ", ".join(available_policies())
        raise ValueError(
            f"Unknown decision policy '{name}'. Available policies: {available}."
        )
    return canonical


def create_policy(name, **parameters):
    """Instantiate one registered decision policy."""

    canonical = canonical_policy_name(name)
    return _POLICY_CLASSES[canonical](**parameters)


def available_policies():
    """Return registered canonical names in deterministic order."""

    return tuple(sorted(_POLICY_CLASSES))
