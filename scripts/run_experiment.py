import atexit
import json
import os.path as osp
import signal
import time

import docker as _docker
import hydra
from dotenv import load_dotenv
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from agent_scaling.config.run import RunConfig
from agent_scaling.exp_runner import ExperimentRunner
from agent_scaling.logger import add_sink, logger
from agent_scaling.utils import get_run_conf_dir, write_yaml

import litellm
import logging
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

load_dotenv(override=True)


def _cleanup_all_containers():
    """Kill and remove benchmark containers created during experiment runs."""
    try:
        client = _docker.from_env(timeout=10)
        containers = client.containers.list(filters={"status": "running"})
        for c in containers:
            image_name = c.image.tags[0] if c.image.tags else ""
            if "swebench/" in image_name or "terminalbench/" in image_name or "t-bench/" in image_name:
                try:
                    c.kill()
                    c.remove(force=True)
                except Exception:
                    pass
        client.close()
    except Exception:
        pass


def _signal_handler(signum, frame):
    _cleanup_all_containers()
    raise SystemExit(1)


@hydra.main(
    config_path=get_run_conf_dir(), config_name="run_exp.yaml", version_base=None
)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    hydra_cfg: HydraConf = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    cfg_dict = dict(cfg)
    cfg_dict["save_dir"] = output_dir

    config = RunConfig(**{str(k): v for k, v in cfg_dict.items()})

    # Register cleanup handlers to prevent Docker container leaks
    atexit.register(_cleanup_all_containers)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    logger.info(f"log_langfuse={config.log_langfuse}")

    logger.info(
        f"Running experiment with config: {json.dumps(config.get_run_metadata(), indent=2)}"
    )
    write_yaml(config.get_run_metadata(), osp.join(output_dir, "run_config.yaml"))
    runner = ExperimentRunner(config)
    lptr = add_sink(osp.join(output_dir, "run.log"))
    if config.run_parallel:
        logger.info(f"Running experiment in parallel with {config.num_workers} workers")
        all_metrics = runner.run_parallel(num_workers=config.num_workers)
    else:
        logger.info("Running experiment sequentially")
        all_metrics = runner.run()
    time.sleep(2)  # make sure the logs below show up last
    logger.success(
        f"Experiment completed with metrics:\n{json.dumps(all_metrics, indent=2)}"
    )
    logger.info("Results saved to: " + output_dir)
    logger.remove(lptr)


if __name__ == "__main__":
    main()
