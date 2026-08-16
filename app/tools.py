"""Two small, deterministic tools.

They exist so traces contain real tool spans with real arguments. The RF flavour
is incidental — what matters for the harness is that one of them raises on bad
input, which gives you an error path to look at in the tracing product.
"""

from __future__ import annotations

from typing import Any

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "wifi_channel_info",
        "description": (
            "Return the centre frequency and width of a Wi-Fi channel. Use this "
            "whenever the user names a channel number, instead of recalling the "
            "frequency from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "integer",
                    "description": "Channel number, e.g. 6, 36, 149.",
                },
                "band": {
                    "type": "string",
                    "enum": ["2.4", "5", "6"],
                    "description": "Band in GHz that the channel number belongs to.",
                },
            },
            "required": ["channel", "band"],
        },
    },
    {
        "name": "dbm_to_milliwatts",
        "description": (
            "Convert a power level in dBm to milliwatts. Use this for any dBm "
            "conversion the user asks for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dbm": {
                    "type": "number",
                    "description": "Power level in dBm. May be negative.",
                }
            },
            "required": ["dbm"],
        },
    },
]


class ToolError(Exception):
    """Raised for input the tool cannot serve. Surfaces as an error tool result."""


def _wifi_channel_info(channel: int, band: str) -> str:
    if band == "2.4":
        if channel == 14:
            centre = 2484
        elif 1 <= channel <= 13:
            centre = 2412 + (channel - 1) * 5
        else:
            raise ToolError(f"Channel {channel} is not valid in the 2.4 GHz band (1-14).")
        width = 20
    elif band == "5":
        if not (32 <= channel <= 177):
            raise ToolError(f"Channel {channel} is not valid in the 5 GHz band (32-177).")
        centre = 5000 + channel * 5
        width = 20
    elif band == "6":
        if not (1 <= channel <= 233):
            raise ToolError(f"Channel {channel} is not valid in the 6 GHz band (1-233).")
        centre = 5950 + channel * 5
        width = 20
    else:
        raise ToolError(f"Unknown band {band!r}; expected '2.4', '5' or '6'.")

    return (
        f"Channel {channel} in the {band} GHz band has a centre frequency of "
        f"{centre} MHz and a default width of {width} MHz "
        f"(occupying {centre - width // 2}-{centre + width // 2} MHz)."
    )


def _dbm_to_milliwatts(dbm: float) -> str:
    milliwatts = 10 ** (float(dbm) / 10)
    return f"{dbm} dBm = {milliwatts:.6g} mW."


_DISPATCH = {
    "wifi_channel_info": _wifi_channel_info,
    "dbm_to_milliwatts": _dbm_to_milliwatts,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool. Raises ToolError for anything the caller got wrong."""
    fn = _DISPATCH.get(name)
    if fn is None:
        raise ToolError(f"No such tool: {name}")
    try:
        return fn(**arguments)
    except ToolError:
        raise
    except TypeError as exc:
        raise ToolError(f"Bad arguments for {name}: {exc}") from exc
