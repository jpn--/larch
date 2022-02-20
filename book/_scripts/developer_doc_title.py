import os
from pathlib import Path
import sys

from ruamel.yaml import YAML

config_file = Path(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "_config.yml",
    )
)

yaml = YAML(typ="rt")  # default, if not specfied, is 'rt' (round-trip)
content = yaml.load(config_file)
content["title"] = sys.argv[-1]
yaml.dump(content, config_file)
