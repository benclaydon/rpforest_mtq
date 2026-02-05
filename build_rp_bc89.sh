source .venv/bin/activate

pip install "cython>=0.29.30"

python setup.py cythonize

pip install -e .