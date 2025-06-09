from minerva.pipelines.experiment import Experiment


def main():
    from jsonargparse import CLI

    result = CLI(Experiment, as_positional=False)
    print("Experiment completed successfully!")
    print(f"Result: {result}")
    print("✨ 🍰 ✨")


if __name__ == "__main__":
    main()
