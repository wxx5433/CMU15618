#!/bin/bash
## spawner.sh
CMD=$0
NUM_WORKERS=$1
TRACE=$2
JOBNAME=$PBS_JOBID

cd $PBS_O_WORKDIR
PATH=$PATH:$PBS_O_PATH

MASTER=`head -n1 $PBS_NODEFILE`
NODES=`sort -u $PBS_NODEFILE`

if [ $MASTER.local = $HOSTNAME ] ; then
  echo $LD_LIBRARY_PATH
  echo "spawning master process on $HOSTNAME ($PBS_NODENUM of $PBS_NUM_NODES)"
  echo $NODES

  echo "command was: '$CMD $NUM_WORKERS $TRACE'"

  . ~/.bashrc
  . ~/.bash_profile
  module load opt-python
  module load gcc-4.9.2

  # make a logs directory
  mkdir -p logs.$JOBNAME

  # Start the launcher and sleep a moment to make sure it is listening.
  ./scripts/nodemanager_local.py --nodefile $PBS_NODEFILE 6666 --log_dir=logs.$JOBNAME $debug_cflags $configflags &

  nodemanager_pid=$!
  sleep .5

  # Start the master and sleep a moment to make sure it is listening.
  ./master --max_workers $NUM_WORKERS --address=$(hostname):15418 --log_dir=logs.$JOBNAME $debug_cflags $(hostname):6666 &
  master_pid=$!
  sleep .5

  # Run the trace driver
  ./scripts/workgen.py $debug_pyflags $(hostname):15418 $TRACE

  success=$?

  # Tell the master to die by sending it the tagged message (shutdown, 0).
  #python -c 'import comm; import sys; sys.stdout.write(comm.TaggedMessage(comm.SHUTDOWN, 0).to_bytes())' | nc $(hostname) 15418
  printf "\x06\x00\x00\x00\x00\x00\x00\x00" | nc $(hostname) 15418
  trap "kill -9 $master_pid; kill -9 $nodemanager_pid; exit $success" SIGINT
  wait $master_pid

  # Kill the the launcher to clean up.
  kill -9 $nodemanager_pid

  # Return successfully iff the all work was accepted correctly.
  exit $success
fi
