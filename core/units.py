
class Unit():
    def __init__(self, **kwargs):
        for key, value in units.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
        self.__dict__.update(kwargs)


class SpatialUnits():
    def __init__(self, ):
        pass
    meters = 'm'
    centimeters = 'cm'
    millimeters = 'mm'
    micrometers = 'um'
    nanometers = 'nm'
    picometers = 'pm'
    femtometers = 'fm'
    attometers = 'am'
    inches = 'in'
    feet = 'ft'
    yards = 'yd'
    miles = 'mi'
    nautical_miles = 'nmi'
    astronomical_units = 'au'
    light_years = 'ly'
    parsecs = 'pc'

class EventUnits():
    def __init__(self, ):
        pass
    events = 'ev'
    spikes = 'spk'
    

class TimeUnits():
    def __init__(self, ):
        pass

    hours = 'h'
    minutes = 'min'
    seconds = 's'
    milliseconds = 'ms'
    microseconds = 'us'
    nanoseconds = 'ns'
    picoseconds = 'ps'
    femtoseconds = 'fs'
    attoseconds = 'as'

class SpeedUnits():
    def __init__(self, ):
        pass
    meters_per_second = 'm/s'

    kilometers_per_hour = 'km/h'
    miles_per_hour = 'mph'
    knots = 'kn'
    mach = 'Ma'

class AngleUnits():
    def __init__(self, ):
        pass
    radians = 'rad'
    degrees = 'deg'

class RateUnits():
    def __init__(self, ):
        pass
    hertz = 'Hz'
    kilohertz = 'kHz'
    megahertz = 'MHz'
    gigahertz = 'GHz'
    terahertz = 'THz'
    petahertz = 'PHz'
    exahertz = 'EHz'
    zettahertz = 'ZHz'
    yottahertz = 'YHz'

class ElectricUnits():
    def __init__(self, ):
        pass
    volts = 'V'
    millivolts = 'mV'
    microvolts = 'uV'
    nanovolts = 'nV'
    picovolts = 'pV'
    femtovolts = 'fV'
    attovolts = 'aV'

