#!/bin/bash -e
REQUIRED_VARS=(
    ARGS
    TRAVIS_BRANCH
    DOCKERFILE
    DOCKER_PROJECT
    DOCKER_IMAGE_TAG
)


help(){
    cat <<EOH
    Usage: $0 [options]
    Options:
        -h|--help
            Show this help
        -v|--verbose
            Enable verbose mode
    Required environment variables:
        $(for var in "${REQUIRED_VARS[@]}"; do echo -ne "$var\n        "; done)
EOH
    exit "${1:-0}"
}


retry() {
    if ! "$@"; then
        echo "******** Retrying $*"
        "$@" || exit $?
    fi
}


parse_options(){
    local opts
    opts=$(\
        getopt \
            --options 'hv' \
            --longoptions 'help,verbose' \
            --name "$0" \
            -- "$@" \
    ) ||  help 1
    local failed=false
    local env_var
    eval set -- "$opts"
    while true; do
        opt="$1"
        shift;
        case $opt in
            -h|--help) help 0;;
            -v|--verbose) set -x;;
            --) break;;
        esac
    done
    for env_var in "${REQUIRED_VARS[@]}"; do
        [[ -v $env_var ]] || {
            echo "Undefined var $env_var, please define befor running this" \
                "script"
            failed=true
        }
    done
    if $failed; then
        echo -e "Some required env vars were not defined, aborting\n"
        help 1
    fi
}


main(){
    parse_options "$@"
    # fail on unset variables expansion
    set -o nounset

    if [[ "$TRAVIS_BRANCH" == "master" ]]; then
        TAG="$DOCKER_PROJECT:$DOCKER_IMAGE_TAG"
    else
        TAG="$DOCKER_PROJECT:dev.$TRAVIS_BRANCH-$DOCKER_IMAGE_TAG"
    fi

    echo "Building image $TAG"
    retry docker build -f "$DOCKERFILE" $ARGS -t "$TAG" .
}


main "$@"