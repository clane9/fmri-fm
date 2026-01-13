import argparse
import logging
import time

from lightning_sdk import Job, Status

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


def resubmit(name: str, max_retries: int | None = None, delay: int = 60):
    job = Job(name=name)
    retries = 0
    while True:
        if job.status == Status.Stopped:
            if max_retries and retries >= max_retries:
                logging.info("max number of retries reached; exiting")
                return

            job = Job.run(
                name=name,
                machine=job.machine,
                command=job.command,
                studio=job.studio,
                interruptible=True,
            )
            retries += 1
            logging.info(f"resubmitted {name} ({job.name}) (n={retries})")

        elif job.status in {Status.Completed, Status.Failed}:
            logging.info(f"job exited with status {job.status}; exiting")
            return

        time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="job name")
    parser.add_argument("--max-retries", "-n", type=int, default=None, help="max number of retries")
    args = parser.parse_args()

    resubmit(args.name, args.max_retries)
