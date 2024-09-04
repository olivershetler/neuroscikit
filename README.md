# neuroscikit
This is the initial branch of the main pipeline for data analysis at the Hussaini Lab. It will eventually be forked to Hussaini Lab's GitHub account when it's ready to be used. This repository is designed to be collaborateively developed according to the guidelines laid out below.

## Architecture

There are four main software modules called `core`, `library`, `scripts` and `widgets`. There are also several additional bridge / firmware modules that are all named by the convention `x_{}` (e.g. `x_io`, `x_cli`, etc.).

### io (input/output)
In the io layer, there are submodules for reading and writing different data formats, and tools for getting data from users. Additionally, there are two modules for creating the data classes Session and Study. These classes serve as a bridge and formatting bottleneck between the io layer and the core data types. They store data in a in a standardized format that is easy to convert into the core data types. This format is isomorphic to dictionaries with keys and contents as outlined below (i.e. every io module must generate a subset of the following keys and contents).

#### Session class intake dictionary

- animal
    - animal_id
    - species
    - sex
    - age
    - weight
    - genotype
    - animal_notes
- devices
    - implants
        - implant_id
        - implant_type
            - tetrode
            - tetrode_array_{int}
            - sillicone_shank
            - ...
        - implant_geometry
            - channel_{int}: (x, y, z)
            - ...
        - implant_data
            - sample_rate OR irregular_sampling: True
            - event_times: [float]
            - event_labels: [string]
            - channel_{int}: [float]
            - units: string
            -
    - axona_led_tracker
        - led_tracker_id
        - led_location
        - led_position_data: {`time`:float, `x`:float, `y`:float}

#### Study class intake dictionary


### core
The `core` module contains the canonical data types and functions for managing them. This module has no dependencies other than the version of python being used and numpy. Some of the data structures are inspired by the [Neurodata Without Borders](https://www.nwb.org/) project, though the code is not directly based on that project.

### library
The `library` module contains data types, classes and functions for exploring, analyzing and visualizing the core data types. Each of these functions does only one thing.

### scripts
The `scripts` module contains use cases where settings are configured up front and then several functions are called in sequence to perform the desired task. There are three main types of scripts. (1) Batch processing scripts that run on a large number of files. (2) Parameter exploration / optimization scripts that run on a single file. (3) Automations for smaller tasks that require multiple library functions, are repeated often, and should be performed the same way everywhere. The third type of script may be called by the first two.

### widgets
The `widgets` module contains framework-agnostic widgets for interactive analysis. Each widget is constituted of two parts. The first part is a `View` class that contains all the states of the widget, as well as any text that would be displayed to the user. The second part is a `Controller` class that contains all the logic for the widget. The `Controller` class is responsible for updating the `View` class when the user interacts with the widget and for calling the appropriate functions in the `library` or `scripts` module. Each widget is structured as a microservice, so there are no dependencies between the widgets. Any information that is needed by a widget is passed as a parameter to the `Controller` class. Communication between two widgets (if ever necessary) would be done through separate bridge submodules in the `widgets` module.

### x_{} modules

The `x_{}` modules contain interfaces for frameworks and APIs. Moreover, they contain adapters, gateways and other glue code for the core and library modules.

The `x_cli` module contains the command line interfaces for running scripts and launching widgets.

The `x_gui` module bridges GUI framework(s) for the widgets.

The `x_io` module contains the functions for reading and writing data from various formats in various frameworks (e.g. databases, os file systems, server APIs, etc.).


## Guidelines for Contributors

If you would like to contribute to the project, please read all the guidelines first.

### Style

- Generally follow the [PEP 8](https://peps.python.org/pep-0008/) style conventions.
- Additionally, please name functions and classes in such a way that there are no names that are substrings of other names. For example, `get_waveform` and `get_waveform_from_file` are not allowed. Instead, use something like `get_waveform` and `waveform_from_file_path`. This is to allow global search and replace when changing the name of a function or class.
- When importing modules:
    - Group all the internal dependencies together at the top under the header "# Internal Dependencies".
    - Group all the external dependencies together below header the internal dependencies.
        - Divide them into subgroups under the headers "# A+ Grade External Dependencies," "# A Grade External Dependencies" and "# Other External Dependencies".
        - See the [External Dependencies](#external-dependencies) section for more details on the lists.

### Test Driven Development (TDD)
- TDD is **required** for contributors.
- Use `pytest` for unit tests.
- Store test files for a given module (top folder level) in a folder of the form `{module}_tests`.
- If there are folder level submodules, create a subfolder named with the pattern `{submodule}_tests`.
- For each submodule in the module, create a test file and name it `test_{submodule}.py`.
**This convention is very important because it allows `pytest` to find the tests.**
- Write your test(s) before writing your new function or class.
- A function or class should either have an automated test or contain **only** code that requires manual testing.
    - e.g. `View` and `Controller`classes in the `widgets` module are segregated from the `Window` classes in the `x_gui` module;
    - the `Window` classes do exactly and only two things---display the state of a `View` object and feed user input to the `Controller`.

### Architectural Dependencies
- Dependencies between layers flow only one way:  `core` < `library` < `scripts` < `widgets` < `x_{}`.
    - e.g. `library` calls only `core`, `scripts` can directly call `library` or `core`, ect.
- Limit interdependency among modules in the same layer. Segregate large scripts or widgets into microservices that are independent of each other (no common databases or global states).
- For function modules, create roughly one public function per module. Name helper functions using an underscore as the first character (e.g. `_helper_function`).
- For class modules, use only a handfull of abstract classes per module (preferably no more than one).

### External Dependencies
- `core` should not depend on any external libraries except numpy.
- `library` should depend only on `core` and a few A+ safe libraries (see below).
- `scripts` and `widgets` should depend on `library`, `core` and A safe libraries.
- The `x_{}` modules can have dependencies on any framework or firmware that is reasonably well maintained.
    - Whenever possible, a submodule in a given `x_{}` module (e.g. `x_gui.pyside6`) should depend only on the framework listed.
    - Exceptions exist. For example, the `x_cli` module contains submodules that launch widgets using dependencies on `x_gui` submodules. When doing this, try to keep the dependencies as minimal as possible.

#### A+ Safe Libraries
- numpy
- pandas
- matplotlib

#### A Safe Libraries
- scipy
- statsmodels
- plotly
- PIL / pillow

#### Current Framework / Bridge Dependencies
- Qt / PySide6
- Axona / TINT file formats (.set, .bin, .X, .cut, .pos, .eeg, .egf)
    - .X file extensions denote files where there is an integer after the dot (e.g. .1, .2, .10, etc.)
- Intan file formats (.rhd, .rdh?)

If you'd like to propose adding a library to the A Safe or A+ Safe list, please contact the current maintainer [Oliver Shetler]() at [cos2112@cumc.columbia.edu](cos2112@cumc.columbia.edu).

### Git Ettiquette
- atomic commits---commit every change to every function whenever the relevant test is passing
	- only commit the relevant module (and test)
- frequent commits---DO NOT build an entire feature and then commit. This can lead to merge conflicts
- frequent pulls---pull every time before committing and notify partner if there is a merge conflict (tell them to pause on that module)
- separate hard to test parts of a class or function from the easy parts

### Contributing to specific modules

#### core
`core` is the most fundamental module. It should not depend on any other module. It should only contain classes and functions that pertain to basic data types that could conceivably have been read from a file.

When defining new class, please consider the following:
- Metadata: Are the data in the class metadata for an object that was used in the experiment?
    - If yes, then the class should be in `core`.
    - If no, then the class might belong in `library`.
- Unit Homogenaity: Does the class contain an attriute that contains one principle data structure? Are all the data within the principle attribute of the same unit type?
    - If yes to both, then the class should be in `core`.
    - If no, then the class might belong in `library`.

When defining a new function, please consider the following:
- Does the function operate on a single data type with homogeneous units?
    - If yes, then the function should be in `core`. It might belong inside the pertainant class.
    - If no, then the function might belong in `library`.

#### library
`library` is the module that contains the bulk of the code. It should depend only on `core` and a few A and A+ safe libraries (see above). It contains the data structures and functions that are used to analyze, visualize and derive new data types from the data objects in the `core` module.

When defining new classes, please consider the following:
    - Does the class combine data from multiple `core` classes?
        - If yes, then the class should be in `library`.
        - If no, then the class might belong in `core`.
    - Does the class or function perform a non-invertable transformation on a single data type and/or does the operation change the units? (e.g. converting locations to speeds or velocities)
        - If yes, then the class should be in `library`.
        - If no, then the class might belong in `core`.
    - Does the class store a single object for a single task?
        - If yes, then the class should be in `library`.
        - If no, then the class might belong in `scripts` or `widgets`.
    - Does the class contain only methods that do not cause side effects? (e.g. no direct manipulation of data; only creation and deletion of data attributes; e.g. no read/write or display methods)
        - If yes, then the class should be in `library`.
        - If no, then the class might belong in `scripts` or `widgets`.

When defining new functions, please consider the following:
    - Does the function perform a single analysis, reformatting or visualization task (without displaying)?
        - If yes, then the function should be in `library`.
        - If no, then the function might belong in `scripts` or `widgets`.

#### scripts
Architecture rules:
- Horizontal Segregation: Scripts should not depend on each other.
- Vertical Segregation: Scripts should not depend on `widgets` or any `x_{}` modules.
- No Global States: Scripts should not have any global states.
- No Interfaces: Scripts should not contain any adapters or interfaces.
- One Configuration Dictionary: Scripts should always take in exactly ONE configuration dictionary.
- Data Structures: Aside from one configuration dictionary, scripts should only take in data structures from the `core` and `library` modules.

When defining new classes or functios in the `scripts` module, please consider the following:
    - Does the class or function perform a sequence of steps that automate a use case?
        - If yes, then the class or function should be in `scripts`.
        - If no, then the class or function might belong in `library` or `widgets`.
    - Does the class or function require user input only at the beginning of the use case?
        - If yes, then the class or function should be in `scripts`.
        - If no, then the class or function might belong in `widgets`.

#### widgets
When defining new classes in the `widgets` module, please consider the following:
    - Does the class store part of an abstract user interface (e.g. a `View` or `Controller`)?
        - If yes, then the class should be in `widgets`.
        - If no, then the class might belong in `scripts` or `library`.
    - Does the class contain

