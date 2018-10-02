#!/bin/bash -ev

conf_python_base_versioned() {
    export DOCKER_PROJECT=vbalbp/inspire-classifier
    export DOCKER_IMAGE_TAG=3.6
    export DOCKERFILE=Dockerfile
    export ARGS='--build-arg=INSPIRE_PYTHON_VERSION=3.6'
}

conf_python_base_versioned; ./build.sh

conf_python_base_versioned; ./deploy.sh
