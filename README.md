# Implementation of a Bespoke, Collaborative-Filtering, Matrix-Factoring Recommender System Based on a Bernoulli Distribution Model for Webel
This is my Senior Year Project, presented on the 7th of July at Universidad Politécnica de Madrid, to obtain my Software Engineering Degree (Undergraduate/Bachelor's).

It is the implementation and practical use of a Machine Learning Recommender System ([Bernoulli Matrix Factorisation](https://github.com/ferortega/bernoulli-matrix-factorization), or BeMF) for [Webel](https://appwebel.com/), using a Jupyter Notebook. It encompasses the fields of Data Science, Machine Learning and Software Engineering, applied to a real-world scenario.

## Tool installation and environment setup

Installing the necessary libraries depends on the target device’s OS. Everything shown will be done for a **Microsoft-based PC**, although the following steps can also be done on **any Linux distribution or MacOS device**.

Python 3.6 or greater is required for the following process. To install, simply navigate to [the Python website](https://www.python.org/downloads/) and download one of the suitable versions.

Next, open a Terminal and navigate to the folder where the project was downloaded. The necessary modules can be installed by creating and activating a new virtual environment and thanks to the `requirements.txt` file:

```terminal
python -m venv TFGEnv
pip install -r ./requirements.txt
```

To activate the newly created virtual environment, run `./TFGEnv/Scripts/Activate.ps1` for PowerShell or `./TFGEnv/Scripts/activate.bat` if using cmd to activate the aforementioned virtual environment (`source TFGEnv/bin/activate` for MacOS).

With that, the virtual environment should now be active, containing all the required modules. Simply run `jupyter notebook --autoreload` and a tab should open in your browser. To open the notebook, click on the .ipynb file and the project will open.

**NOTE**: For NDA reasons I had to remove certain columns as well as the original data, so do not expect to be able to replicate this project easily in your own device. I'm sorry for the inconvenience but this was the only way I would be allowed to share this work.
