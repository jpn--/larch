import os
import setuptools_scm

def get_version():
    version = setuptools_scm.get_version(root='..', relative_to=__file__)
    os.environ["LARCH_VERSION"] = version
    v_file = os.path.join(os.path.dirname(__file__), "..", "LARCH_VERSION.txt")
    with open(v_file, 'wt') as f:
        f.write(version)
    print(f"LARCH_VERSION = {version}")
    return version

if __name__ == '__main__':
    get_version()
