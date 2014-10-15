z:
cd Larch
python setup.py build_ext --swig=c:\swigwin-3.0.2\swig.exe
python setup.py install --home=.\wayground
python -i ./wayground/wplay.py
cmd /k