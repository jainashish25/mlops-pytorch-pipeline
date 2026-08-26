# Reflection

## What was the most challenging part?

By a wide margin, the most challenging part was not the machine learning or even
writing the Kubernetes manifests — it was getting a **GPU to be schedulable
inside a Kubernetes pod on a Windows + WSL2 machine**. The model, the
Dockerfiles, and the YAML came together quickly; the infrastructure underneath
them fought back at almost every layer.

The GPU had to be visible through four nested layers, and each was a separate
battle. It worked in native Windows Python immediately, but that counts for
nothing inside Kubernetes. Getting it into WSL2 was fine (the Windows NVIDIA
driver passes through automatically), and getting it into a plain Docker
container also worked. The layer that broke was the last one: **GPU inside
Minikube's container**. The `nvidia-device-plugin` pod kept logging
`Failed to initialize NVML: Not Supported`, and the node refused to advertise
`nvidia.com/gpu`, so every GPU-requesting pod would have stayed `Pending`.

The root cause turned out to be **Docker Desktop**. Its WSL2 GPU model exposes
the GPU through a DirectX bridge that a top-level `docker run --gpus all` can
use, but that access does not propagate into the nested "container-in-a-
container" that Minikube's `docker` driver creates. The fix was invasive:
remove Docker Desktop's involvement, install the **native Docker Engine inside
WSL2**, add the NVIDIA Container Toolkit, and — the key step — set the NVIDIA
runtime as Docker's *default* runtime with
`nvidia-ctk runtime configure --set-as-default`. Only then, after recreating the
Minikube cluster, did the device plugin initialize NVML and the node report
`nvidia.com/gpu: 1`.

Debugging this was hard precisely because the failure was silent and several
layers removed from the symptom. The pod was `Running`, not crashing; the error
was buried in its logs; and `docker run --gpus all` working at the top level
created a false sense that the GPU was "available" when the cluster still
couldn't see it. The lesson I took away is that in MLOps, *the environment is
the hard part*. Reproducibility has to be verified at every layer of the stack,
not assumed from the layer above — "the GPU works in Docker" and "the GPU works
in a Kubernetes pod" are genuinely different statements.

A smaller but related challenge was **Docker build reliability**: the training
image originally pulled Python 3.11 from an external PPA, which timed out
against Launchpad mid-build. Switching to the base image's stock Python 3.10
removed an entire network dependency and made the build self-contained — a
reminder that every external call in a Dockerfile is a point of failure.

Once the GPU was schedulable, the rest worked as designed: the training Job
writing a checkpoint to a PVC, the serving Deployment reading it, the probes
gating traffic, and the HPA scaling on load. The pipeline reaching 86%
validation accuracy on GPU and serving live predictions was satisfying, but the
real learning was in the plumbing beneath it.
