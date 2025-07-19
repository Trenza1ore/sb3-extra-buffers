pip uninstall sb3_extra_buffers
python setup.py install
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*.whl
twine upload --skip-existing dist/*.tar.gz
REM As of 2023, egg files are no longer accepted for PyPI uploads
