executable = env_install+update.sh
arguments = "-n SADTech ../environment.yml"
getenv = True
error = condor_output/cpu_env_update_install.err
log = condor_output/cpu_env_update_install.log
output = condor_output/cpu_env_update_install.out
notification = complete
transfer_executable = False
request_cpus = 1
queue
