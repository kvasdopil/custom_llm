#!/usr/bin/env python3
"""
Moon weather tool implementation for DeepSeek R1 LangGraph Agent
"""
from langchain.tools import tool
from pydantic import BaseModel, Field


class MoonCoordinatesInput(BaseModel):
    """Input schema for the moon_weather tool."""
    latitude: float = Field(description="The latitude on the moon (degrees)")
    longitude: float = Field(description="The longitude on the moon (degrees)")


@tool("moon_weather", args_schema=MoonCoordinatesInput, return_direct=True)
def moon_weather(latitude: float, longitude: float) -> str:
    """Determine the weather conditions on the moon at specific coordinates.

    The moon is tidally locked to Earth, meaning one side always faces Earth.

    Args:
        latitude: The latitude on the moon in degrees (-90 to 90)
        longitude: The longitude on the moon in degrees (-180 to 180)

    Returns:
        A string describing the weather conditions at the specified location.
    """
    # Validate input ranges
    if not -90 <= latitude <= 90:
        return f"Error: Latitude must be between -90 and 90 degrees. Got {latitude}"

    if not -180 <= longitude <= 180:
        return f"Error: Longitude must be between -180 and 180 degrees. Got {longitude}"

    if -90 <= longitude <= 90:
        description = f"On the Earth-facing side of the moon at coordinates {latitude}°N, {longitude}°E. "
        description += f"With no atmosphere to moderate temperatures, the surface experiences extremely hot (up to 127°C/260°F) temperatures."

        return description
    else:
        # Dark side (far side)
        description = f"On the far side of the moon at coordinates {latitude}°N, {longitude}°E. "
        description += "This area never faces Earth. Without an atmosphere, temperatures are extremely cold (down to -173°C/-280°F)."

        return description
