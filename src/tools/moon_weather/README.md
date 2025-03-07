# Moon Weather Tool

This tool provides simulated weather information for different locations on the moon based on the provided coordinates.

## Files

- `__init__.py`: Exports the tool and related functions
- `tool.py`: Contains the implementation of the moon_weather tool
- `prompt.py`: Contains the prompt template with instructions for using this tool

## Usage

The moon weather tool takes latitude and longitude coordinates as input to determine the weather conditions on the moon:

- Latitude: Value between -90 (South pole) and 90 (North pole)
- Longitude: Value between -180 and 180
  - The Earth-facing side (near side) is roughly between -90 and 90 longitude
  - The far side (dark side) is longitude values outside that range

Example input:

```json
{
  "latitude": 40.0,
  "longitude": 150.0
}
```

## Features

The tool differentiates between:

1. **Earth-facing side**: The side of the moon that always faces Earth
2. **Far side**: The side of the moon that never faces Earth

Each side has different descriptions but similar temperature ranges, simulating the actual conditions on the moon:

- Lunar daytime: Extremely hot (up to 127°C/260°F)
- Lunar nighttime: Extremely cold (down to -173°C/-280°F)

## Modifying or Extending

To modify the tool's functionality:

1. Edit the implementation in `tool.py`
2. Update the prompt instructions in `prompt.py` if necessary

Possible extensions include:

- Adding more detail about moon geography
- Including information about lunar phases
- Adding information about specific lunar features at the coordinates
