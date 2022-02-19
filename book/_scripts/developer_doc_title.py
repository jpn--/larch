import os
from pathlib import Path

from ruamel.yaml import YAML

import sharrow as sh

config_file = Path(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "_config.yml",
    )
)

yaml = YAML(typ="rt")  # default, if not specfied, is 'rt' (round-trip)
content = yaml.load(config_file)
content["title"] = f"PRE-RELEASE DEV DOCS"
yaml.dump(content, config_file)
