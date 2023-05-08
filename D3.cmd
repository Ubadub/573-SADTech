executable = src/d3_run.sh
getenv = True
error = condor_output/D3/err.err
log = condor_output/D3/log.log
output = condor_output/D3/out.out
notification = complete
transfer_executable = False
queue
