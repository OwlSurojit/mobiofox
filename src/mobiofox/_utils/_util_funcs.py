from functools import wraps

from qtpy.QtCore import QTimer


def extract_unit_and_value(pixelsize: float) -> tuple[str, float]:
    """
    Convert a pixel size value to a human-readable unit and scale it accordingly.

    Args:
        pixelsize (float): The pixel size in meters.

    Returns:
        tuple[str, float]: A tuple containing the unit and the scaled pixel size.
    """
    if pixelsize < 1e-9:
        unit = "pm"
        pixelsize *= 1e12
    elif pixelsize < 1e-6:
        unit = "nm"
        pixelsize *= 1e9
    elif pixelsize < 1e-3:
        unit = "µm"
        pixelsize *= 1e6
    elif pixelsize < 1:
        unit = "mm"
        pixelsize *= 1e3
    else:
        unit = "m"

    return unit, pixelsize


def get_unit_factor(unit: str) -> float:
    """
    Get the conversion factor for a given unit.

    Args:
        unit (str): The unit to convert from.

    Returns:
        float: The conversion factor for the unit.
    """
    if unit == "pm":
        return 1e-12
    elif unit == "nm":
        return 1e-9
    elif unit == "µm":
        return 1e-6
    elif unit == "mm":
        return 1e-3
    elif unit == "m":
        return 1.0
    else:
        raise ValueError(f"Unknown unit: {unit}")


def debounce(wait_time_ms):
    """
    A decorator to debounce a method using a QTimer.

    Parameters:
        wait_time_ms (int): The debounce time in milliseconds.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "_debounce_timers"):
                self._debounce_timers = {}
            if func not in self._debounce_timers:
                self._debounce_timers[func] = QTimer()
                self._debounce_timers[func].setSingleShot(True)
            else:
                self._debounce_timers[func].timeout.disconnect()
            self._debounce_timers[func].timeout.connect(
                lambda: func(self, *args, **kwargs)
            )
            self._debounce_timers[func].start(wait_time_ms)

        return wrapper

    return decorator
