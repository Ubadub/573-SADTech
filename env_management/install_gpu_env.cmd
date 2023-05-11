executable = env_install+update.sh
arguments = "-n gpu_SADTech ../gpu_environment.yml"
getenv = True
error = condor_output/gpu_env_update_install.err
log = condor_output/gpu_env_update_install.log
output = condor_output/gpu_env_update_install.out
notification = complete
transfer_executable = False
request_GPUs = 1
Requirements = (Machine == "patas-gn2.ling.washington.edu")
queue
