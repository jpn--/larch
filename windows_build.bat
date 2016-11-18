L:
git pull
set OPENBLAS_NUM_THREADS=1
python update_git_version.py
python setup.py build_ext
python setup.py install --user
if %ERRORLEVEL% NEQ 0 (
  REM Install failed
  echo python install larch returned error %ERRORLEVEL%
  EXIT /B 1
)
rm dist/*.whl
python setup.py bdist_wheel
EXIT /B %ERRORLEVEL%
