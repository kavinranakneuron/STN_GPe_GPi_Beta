"""bgnet: spiking basal-ganglia network framework (JNE-110355 rebuild).

Unit conventions throughout the package (current density):
    conductance: mS/cm^2
    current:     uA/cm^2
    voltage:     mV
    time:        ms
    calcium:     uM
    capacitance: uF/cm^2

No scaling factors anywhere. If a literature source uses different units,
convert at the citation point in the source file.
"""

__version__ = "0.1.0"
