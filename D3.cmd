executable = src/d3_run.sh
getenv = True
error = condor_output/D3/D3.err
log = condor_output/D3/D3.log
output = condor_output/D3/D3.out
notification = complete
transfer_executable = False
request_GPUs = 1
Requirements = (Machine == "patas-gn2.ling.washington.edu")
queue
