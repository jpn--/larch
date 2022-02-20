import os
import setuptools_scm
import sys

def get_version():
    version = setuptools_scm.get_version(root='..', relative_to=__file__)
    os.environ["LARCH_VERSION"] = version
    if len(sys.argv) >= 2:
        v_file = sys.argv[1]
    else:
        v_file = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "LARCH_VERSION.txt")
        )
    with open(v_file, 'wt') as f:
        f.write(f"LARCH_VERSION={version}\n")
    print(f"LARCH_VERSION={version}")
    return version

if __name__ == '__main__':
    get_version()
