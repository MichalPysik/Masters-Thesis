def format_timestamp(timestamp: float) -> str:
    """
    Format a timestamp in seconds to a human-readable format.

    Args:
        timestamp (float): The timestamp in seconds.

    Returns:
        str: The formatted timestamp in the format "HH:MM:SS.sss".
    """
    minutes, seconds = divmod(timestamp, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
