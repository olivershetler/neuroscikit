# neuroscikit
This is the initial branch of the main pipeline for data analysis at the Husseini Lab. It will eventually be forked to Hussaini Lab's GitHub account when it's ready to be used. This repository is designed to be collaborateively developed according to the guidelines laid out below.

## Architecture

There are four main software modules called `core`, `library`, `scripts` and `widgets`. There are also several additional bridge / firmware modules that are all named by the convention `x_{}` (e.g. `x_io`, `x_cli`, etc.).

### core
The `core` module contains the canonical data types and functions for managing them. This module has no dependencies other than the version of python being used. Some of the data structures are inspired by the [Neurodata Without Borders](https://www.nwb.org/) project, though the code is not directly based on that project.

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
- `core` should not depend on any external libraries.
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

