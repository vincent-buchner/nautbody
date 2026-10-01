class EventMessage:
    def __init__(self, event_type: str, body: str | dict | None = None) -> None:
        self._msg: dict[str, str | dict | None] = {"event_type": event_type}
        self._msg["body"] = body

    def _set_body(self, body: str | dict | None) -> None:
        self._msg["body"] = body

    # NOTE: This means that body is a set only property
    property(fset=_set_body)

    def to_dict(self) -> dict:
        return self._msg
