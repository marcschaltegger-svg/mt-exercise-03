# MT Exercise 3: Layer Normalization for Transformer Models

This repo is a collection of scripts showing how to install [JoeyNMT](https://github.com/joeynmt/joeynmt), download
data and train & evaluate models, as well as the necessary data for training your own model

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3.10 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps for macOS & Linux users

Clone this repository or your fork thereof in the desired place:

    git clone https://github.com/marcamsler1/mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Make sure to install the exact software versions specified in the the exercise sheet before continuing.

Download Moses for post-processing:

    ./scripts/download_install_packages.sh


Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved. It is also possible to continue training from there later on.

To calculate results table and graph:

    ./scripts/extract_and_visualize_results.py

# Steps for Windows users

This repo relies on Bash scripts (.sh files), which do not run natively on Windows (CMD or PowerShell).  
Here are two ways to make it work:

Option 1: Use WSL (Windows Subsystem for Linux)
Enable WSL and install Ubuntu: `wsl --install`

Open Ubuntu from your Start menu.

Inside the Ubuntu terminal, follow the exact same steps as shown above for macOS/Linux:
```
git clone https://github.com/marcamsler1/mt-exercise-03
cd mt-exercise-03
./scripts/make_virtualenv.sh
./scripts/download_install_packages.sh
./scripts/train.sh
```     

Option 2: Manually run steps without shell scripts
If you can't use WSL, you can recreate the process manually using PowerShell or CMD
Create and activate a virtual environment:
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
Manually download and install Moses and other dependencies (you'll need to look inside scripts/download_install_packages.sh to replicate its steps).

Run the training logic by manually executing the code inside train.sh, or porting it to a Python script or notebook.
