import subprocess
import re
import os, sys

if len(sys.argv) < 2:
  print "need the path to the graph"
  sys.exit(1)

path = os.path.abspath(sys.argv[1])
if not os.path.exists(path):
  print " Incorrect path to graphs given"

#init the example job script

username_out = subprocess.Popen(['echo $USER'], shell=True, stdout=subprocess.PIPE).communicate()
username = username_out[0].replace('\n',"")

f = open("example.job", "r")
tmp = open("tmp.job", "w")
for line in f:
  if 'args=' in line:
    tmp.write("args=\'%s\'\n"%(path))
  else:
    tmp.write(line)

tmp.close()
f.close()

subprocess.call("mv example.job old_example.job", shell=True)
subprocess.call("mv tmp.job example.job", shell=True)
subprocess.call("rm tmp.job -f", shell=True)

for i in [1,2,4,8,16,32, 64, 128]:
  subprocess.call("./generate_job.sh %d" % (i), shell=True)
  subprocess.call("qsub %s_%d.job" % (username, i), shell=True)

print "Submission Successful"
print "Deleting all the job scripts"
rm_cmd = "rm -f %s_*" % (username)
subprocess.call(rm_cmd, shell=True)
