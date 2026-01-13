from lightning_sdk import Studio, Machine, Job

studio = Studio(name="connor-pretrain", teamspace="medarc", org="medarc")

# for jobid in range(6):
#     job = Job.run(
#         command=f"bash fmri-fm/experiments/decoders/launch_pretrain.sh {jobid}",
#         name=f"decoders_pretrain_0_{jobid}",
#         machine=Machine.H100,
#         studio=studio,
#         interruptible=True,
#     )
#     print(job.name)

# for jobid in range(2):
#     job = Job.run(
#         command=f"bash fmri-fm/experiments/decoders/launch_pretrain_1.sh {jobid}",
#         name=f"decoders_pretrain_1_{jobid}",
#         machine=Machine.H100,
#         studio=studio,
#         interruptible=True,
#     )
#     print(job.name)

# for jobid in [6]:
#     job = Job.run(
#         command=f"bash fmri-fm/experiments/decoders/launch_pretrain.sh {jobid}",
#         name=f"decoders_pretrain_0_{jobid}",
#         machine=Machine.H100,
#         studio=studio,
#         interruptible=True,
#     )
#     print(job.name)

# for jobid in range(2):
#     job = Job.run(
#         command=f"bash fmri-fm/experiments/decoders/launch_pretrain_1.sh {jobid}",
#         name=f"decoders_pretrain_1_{jobid}",
#         machine=Machine.H100,
#         studio=studio,
#         interruptible=True,
#     )
#     print(job.name)

for jobid in [2]:
    job = Job.run(
        command=f"bash fmri-fm/experiments/decoders/launch_pretrain_1.sh {jobid}",
        name=f"decoders_pretrain_1_{jobid}",
        machine=Machine.H100,
        studio=studio,
        interruptible=True,
    )
    print(job.name)
