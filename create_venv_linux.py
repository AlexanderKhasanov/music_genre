import os

os.system(('python3.9 -m venv venv && '
           '. venv/bin/activate && '
           'pip install -r requirements.txt && '
           'ipython kernel install --user --name=venv_kernel'))
