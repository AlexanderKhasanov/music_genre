import os

os.system(('py -3.9 -m venv venv && '
           'venv\\Scripts\\activate.bat && '
           'pip install -r requirements_win.txt && '
           'ipython kernel install --user --name=venv_kernel'))
