# This is a muti-stage Dockerfile that can be used to build many different types of
# bundled dependencies for PySpark projects. 
# The `base` stage installs generic tools necessary for packaging.
#
# There are `export-` and `build-` stages for the different types of projects.
# - python-packages - Generic support for Python projects with pyproject.toml
# - poetry - Support for Poetry projects
#
# This Dockerfile is generated automatically as part of the emr-cli tool.
# Feel free to modify it for your needs, but leave the `build-` and `export-`
# stages related to your project.
#
# To build manually, you can use the following command, assuming 
# the Docker BuildKit backend is enabled. https://docs.docker.com/build/buildkit/
#
# Example for building a poetry project and saving the output to dist/ folder
# docker build --target export-poetry --output dist .


## ----------------------------------------------------------------------------
##  Base stage for python development
## ----------------------------------------------------------------------------
FROM --platform=linux/amd64 amazonlinux:2 AS base

RUN yum install -y python3 tar gzip

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m pip install --upgrade pip
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$PATH:/root/.local/bin"

WORKDIR /app

COPY . .

# Test stage - installs test dependencies defined in pyproject.toml
FROM base as test
RUN python3 -m pip install .[test]

## ----------------------------------------------------------------------------
##  Build and export stages for standard Python projects
## ----------------------------------------------------------------------------
# Build stage - installs required dependencies and creates a venv package
FROM base as build-python
RUN python3 -m pip install venv-pack==0.2.0 && \
    python3 -m pip install .
RUN mkdir /output && venv-pack -o /output/pyspark_deps.tar.gz

# Export stage - used to copy packaged venv to local filesystem
FROM scratch AS export-python
COPY --from=build-python /output/pyspark_deps.tar.gz /

## ----------------------------------------------------------------------------
##  Build and export stages for Poetry Python projects
## ----------------------------------------------------------------------------
# Build stage for poetry
FROM base as build-poetry
RUN poetry self add poetry-plugin-bundle && \
    poetry bundle venv dist/bundle --without dev && \
    tar -czvf dist/pyspark_deps.tar.gz -C dist/bundle . && \
    rm -rf dist/bundle

FROM scratch as export-poetry
COPY --from=build-poetry /app/dist/pyspark_deps.tar.gz /
