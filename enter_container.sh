#!/bin/bash
# One-command entry into the pp_collision container, with the repo mounted.
# Usage: ./enter_container.sh
docker run -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/projects/PP_collision_clas12:/workspace/PP_collision \
  -w /workspace/PP_collision \
  pp_collision:latest
  