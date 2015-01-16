z:
cd Larch
set OPENBLAS_NUM_THREADS=1
python setup.py build_ext --swig=c:\swigwin-3.0.2\swig.exe
python setup.py install --home=.\wayground
python setup.py bdist_wheel -d wheelhouse
python ./wayground/wtest.py
cmd /k