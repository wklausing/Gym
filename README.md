# Installation Guide

This guide will help you set up the project environment and run your first scenario using `sabreEnv`. Please follow the steps below to ensure a smooth installation.

## Prerequisites

- Python >= 3.10

## Installation Steps

1. Clone the repository into a local folder:

    ```bash
    git clone https://github.com/wklausing/Gym.git <folder>
    ```

2. Change into the cloned directory:

    ```bash
    cd <folder>
    ```

3. Create a Python virtual environment:

    ```bash
    python -m venv .venv
    ```

4. Activate the virtual environment:

    ```bash
    # For Unix or MacOS
    . .venv/bin/activate
    ```

5. Upgrade `pip` to its latest version:

    ```bash
    pip install --upgrade pip
    ```

6. Install the `sabreEnv` package:

    ```bash
    python -m pip install sabreEnv
    ```

## Configuration

- You need to add the package to your Python path. Replace `<dir>` with the path to the directory where `sabreEnv` is located:

    ```bash
    export PYTHONPATH=$PYTHONPATH:<dir>/sabreEnv
    ```

- If you encounter an issue where `sabreEnv` is not found, add the above `export` command to your `.zshrc` or `.bashrc` file, then restart your terminal.

## Running Your First Scenario

1. After completing the installation and configuration, you can run the first scenario:

    ```bash
    python sabreEnv/utils/scenario1.py
    ```

Congratulations! You should now have `sabreEnv` up and running.