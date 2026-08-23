# Jupyter Notebook - Missing Library

CSI 4106 - Fall 2025

Author

Affiliations

Marcel Turcotte [](mailto:marcel.turcotte@uottawa.ca)

School of Electrical Engineering and Computer Science

University of Ottawa

Published

January 30, 2026

# Learning objective

- **Illustrate** the process of identifying and resolving missing library issues in Google Colab.

> **IMPORTANT:**
>
> This example is meant to be executed in [Google Colab](https://colab.research.google.com).

# Import

In this notebook, we use the [langdetect](https://github.com/Mimino666/langdetect) library to accurately identify the language of given text samples.

First, let’s import `langdetect`.

``` python
from langdetect import detect
```

Executing the code cell above will result in an error, as the `langdetect` library is not installed by default in Google Colab.

    ---------------------------------------------------------------------------
    ModuleNotFoundError                       Traceback (most recent call last)
    /tmp/ipython-input-193950075.py in <cell line: 0>()
    ----> 1 from langdetect import detect

    ModuleNotFoundError: No module named 'langdetect'

    ---------------------------------------------------------------------------
    NOTE: If your import is failing due to a missing package, you can
    manually install dependencies using either !pip or !apt.

    To view examples of installing some common dependencies, click the
    "Open Examples" button below.
    ---------------------------------------------------------------------------

This issue can be resolved by adding the following line of code before the first `import` statement. Try it!

``` bash
! pip install langdetect
```

Once this issue has been solved, we can call `detect`. Try it!

``` python
detect("Bonjour tout le monde!")
```

# Best Practice

You could have predicted this scenario. If the `langdetect` library is not installed, captured the resulting exception, and proceeded to install the library.

``` python
try:
    from langdetect import detect
except ImportError:
    print("langdetect not found, installing...")
    import sys
    ! pip install langdetect
    from langdetect import detect  # retry after install
```

# Exploration

`!` allows to run [Unix/Linux shell commands in IPython](https://www.python4data.science/en/latest/workspace/ipython/shell.html). Create a code cell and try these commands.

- `! uname -a` displays information about the system.
- `! ls` displays the content of the current directory.
- `! ls /` displays the content of the root directory.
- `! pwd` returns working directory name.

These commands are useful for debugging code, as they provide information about the computing environment, such as the operating system version and the contents of the local directory.
