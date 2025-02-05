def format_timestamp(timestamp : float) -> str:
    """Format a timestamp in seconds to a human-readable format."""
    minutes, seconds = divmod(timestamp, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"