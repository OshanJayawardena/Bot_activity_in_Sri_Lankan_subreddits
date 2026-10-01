import run_pipeline


def test_pipeline_commands_stay_in_order_and_do_not_run():
    commands = run_pipeline.pipeline_commands(executable="/usr/bin/python", config="config.yaml")

    assert commands == [
        ["/usr/bin/python", "src/collect_arctic.py", "--config", "config.yaml"],
        ["/usr/bin/python", "src/features.py", "--config", "config.yaml"],
        ["/usr/bin/python", "src/text_analysis.py", "--config", "config.yaml"],
        ["/usr/bin/python", "src/report.py", "--config", "config.yaml"],
    ]


def test_run_pipeline_uses_the_injected_runner():
    seen = []

    def runner(cmd, check):
        seen.append((list(cmd), check))

    run_pipeline.run_pipeline(commands=[["python", "src/features.py", "--config", "config.yaml"]], runner=runner)

    assert seen == [(["python", "src/features.py", "--config", "config.yaml"], True)]
