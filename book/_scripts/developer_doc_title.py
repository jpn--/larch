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

if len(sys.argv) >= 2:
    title = sys.argv[1]
else:
    title = "DEVELOPMENT DOCS"
yaml = YAML(typ="rt")  # default, if not specfied, is 'rt' (round-trip)
content = yaml.load(config_file)
content["title"] = title
yaml.dump(content, config_file)
