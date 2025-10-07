import os
import time
import yaml
from minerva.pipelines.experiment import Experiment
from jsonargparse import auto_cli, capture_parser, strip_meta


def get_parser():
    return capture_parser(
        lambda: auto_cli(Experiment, as_positional=False, fail_untyped=True)
    )


def timestamp_now():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def write_experiment_config(experiment, string):
    # create root log directory if it does not exist
    runs_dir = experiment.log_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # write the configuration to a file
    config_file = runs_dir / f"config_{experiment.run_id}.yaml"
    with open(config_file, "w") as f:
        f.write(string)
        print(f"Configuration saved to '{config_file}'")

    return config_file


def write_system_info(experiment):
    # create root log directory if it does not exist
    runs_dir = experiment.log_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # write the system info to a file
    system_info_file = runs_dir / f"system_info_{experiment.run_id}.yaml"
    info = experiment.system_info
    with open(system_info_file, "w") as f:
        yaml.dump(info, f)
        print(f"System info saved to '{system_info_file}'")

    return system_info_file


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = strip_meta(args)
    string = parser.dump(args, format="parser_mode")

    run = parser.instantiate_classes(args).as_dict()
    command_to_run = run.pop("subcommand", None)
    command_to_run_args = run.pop(command_to_run, None)

    run.pop("config", None)
    command_to_run_args.pop("config", None)

    # Instantiate the class with the provided arguments
    experiment = Experiment(**run)

    if command_to_run:
        func = getattr(experiment, command_to_run, None)
        if func is None:
            raise ValueError(
                f"Command '{command_to_run}' not found in {experiment.__class__.__name__}."
            )

        if not callable(func):
            raise ValueError(
                f"Command '{command_to_run}' is not callable in {experiment.__class__.__name__}."
            )

        if command_to_run_args is None:
            command_to_run_args = {}

        is_debug = command_to_run_args.get("debug", False)
        if is_debug:
            print("Running in debug mode. Outputs will not be logged to files.")
            experiment.log_outputs = False
        else:
            print(f"Logging outputs to {experiment.log_dir}")
            write_experiment_config(experiment, string)
            write_system_info(experiment)

        print(f"Running command: {command_to_run} with args: {command_to_run_args}")

        print("\n")
        print("-" * 80)
        print(experiment)
        print("-" * 80)
        print("\n")

        result = func(**command_to_run_args)

        print("Experiment completed successfully!")
        print(f"Result: {result}")
        print("✨ 🍰 ✨")
    else:
        write_experiment_config(experiment, string)
        print("No command provided. Configuration file has been written.")


if __name__ == "__main__":
    main()
