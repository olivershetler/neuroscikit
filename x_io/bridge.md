The bridge module has data classes that form a bottleneck for data intake. They ONLY store data in its most original form (e.g. raw ephys data organized by animal > implant > session > channel, or a spike train that was directly extracted using other software).

                               CORE
x_io -> bridge -> (library -> scripts | widgets) -> x_io | x_gui | etc.