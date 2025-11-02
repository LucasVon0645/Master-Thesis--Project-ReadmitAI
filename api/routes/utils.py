from typing import Dict, List, Any

def _to_dict_list(items) -> List[Dict[str, Any]]:
    out = []
    if items is None:
        return out
    for r in items:
        if hasattr(r, "model_dump"):   # Pydantic v2
            out.append(r.model_dump(exclude_none=True))   # add by_alias=True if you use aliases
        elif hasattr(r, "dict"):       # Pydantic v1
            out.append(r.dict(exclude_none=True))
        else:
            out.append(dict(r))
    return out