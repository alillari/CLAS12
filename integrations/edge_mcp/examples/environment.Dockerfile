# Dockerfile body: the tool inserts FROM using the resolved local base image.
RUN python -c "import sys; print(sys.version)"
ENV EDGE_EXPLORATION=enabled
