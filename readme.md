This is the official implementation of the ICML 2025 submission "Nesterov Method for Asynchronous Pipeline Parallel Optimization".

To run our method, first install requirements.txt and simply run the bash script run.bash.
This script assumes an instance with at least 8 GPUs.

The implementation is based on the pipeline parallelism framework [PipeDream](https://github.com/msr-fiddle/pipedream)
