from typing import Any, Dict


from typing import Dict
import yaml
from typing import Any, Dict
from jsonargparse import ActionConfigFile, ArgumentParser

from ssl_tools.pipelines.base import Pipeline

def get_parser(commands: Dict[str, Pipeline] | Pipeline):
    parser = ArgumentParser()
    
    if isinstance(commands, Pipeline):
        commands = {"run": commands}
        
    subcommands = parser.add_subcommands()

    for name, command in commands.items():
        subparser = ArgumentParser()
        subparser.add_class_arguments(command)
        subparser.add_argument(
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file, in YAML format",
        )
        subcommands.add_subcommand(name, subparser)

    return parser


def auto_main(commands: Dict[str, Pipeline] | Pipeline, print_args: bool = False) -> Any:
    parser = get_parser(commands)

    args = parser.parse_args()
    config_file = args[args.subcommand].pop("config", None)

    if config_file:
        config_file = config_file[0].absolute
        with open(config_file, "r") as f:
            config_from_file = yaml.safe_load(f)

        config = config_from_file
        config.update(args[args.subcommand])
    else:
        config = dict(args[args.subcommand])

    if print_args:
        print(config)

    pipeline: Pipeline = commands[args.subcommand](**config)
    return pipeline.run()
