def decimal_to_dms(lat, lon):
    def convert(value, is_lat=True):
        direction = ('N' if is_lat else 'E') if value >= 0 else ('S' if is_lat else 'W')
        abs_val = abs(value)
        degrees = int(abs_val)
        minutes = int((abs_val - degrees) * 60)
        seconds = (abs_val - degrees - minutes / 60) * 3600
        return f"{degrees}\u00b0{minutes:02d}'{seconds:04.1f}\" {direction}"

    return convert(lat, is_lat=True), convert(lon, is_lat=False)