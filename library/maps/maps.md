

Spatial ratemaps takes a regularly sampled timeseries of positions, spike timestamps and an arena geometry. Additionally, depending upon the method, there will be smoothing and/or binning parameters.

R1: Parent Class / Polymorphism
R2: Operation on or feature of
R3: Darivation / Derived From / Computed From
R4: Attribute / Property / Feature / Nested Class (">")

Classes:
    - Study > Implant(Device)(Animal)() > LED_Signal()(Session)() > Position(Acquisition)()()
    - Study > Implant(Device)(Animal)() > Session > Cell()()(Ephys) > Spike > SpikeTime
    - Study > Implant(Device)(Animal)() > Session > Channel > Ephys(Acquisition)()()


    - Study > Environment > Arena > ArenaGeometry
    - Study > Acquisitions > Camera > Video(Animal)


Acquisition vs Derivative decorator (makes )