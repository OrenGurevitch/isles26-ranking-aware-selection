default: check

# lint + typecheck + test — the gate before any commit
check:
    uv run ruff check frozen_isles/ scripts/ container/
    uv run basedpyright frozen_isles/ scripts/ container/
    uv run pytest frozen_isles/ scripts/ -q

# prove our metric wrapper still equals the organizers' shipped code
verify-metrics:
    uv run pytest frozen_isles/metrics/test.py -v

# stage a trained fold first:
#   uv run python scripts/stage_nnunet_container_weights.py --model <.../TRAINER__PLANS__CONFIG> --folds all
# build the submission container
nnunet-build:
    test -d container/resources_nnunet || (echo "run scripts/stage_nnunet_container_weights.py first" && exit 1)
    docker build --platform=linux/amd64 -f container/Dockerfile.nnunet -t isles26-nnunet:latest .

# run the built container against a simulated Grand Challenge mount
container-run IMAGE INPUT OUTPUT:
    docker run --rm --platform=linux/amd64 --network none \
        -v {{INPUT}}:/input:ro -v {{OUTPUT}}:/output \
        {{IMAGE}}
