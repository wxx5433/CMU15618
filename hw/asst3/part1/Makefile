OBJDIR=objs
SRCDIR=src
TOOLDIR=tools
ifneq (,$(findstring ghc,$(HOSTNAME)))
  # For ghc machines
  CXX=/usr/lib64/openmpi/bin/mpic++
  MPIRUN=/usr/lib64/openmpi/bin/mpirun
  LD_LIBRARY_PATH=/usr/lib64/openmpi/lib
else ifneq (,$(findstring blacklight,$(HOSTNAME)))
  CXX=mpic++
  # Do not run this on blacklight
  MPIRUN=
else
  CXX=mpic++
  MPIRUN=mpirun
endif

mpi:=$(shell which mpic++ 2>/dev/null)
ifeq ($(mpi),)
  $(error "mpic++ not found - did you set your environment variables or load the module?")
endif

SRCS=\
     $(SRCDIR)/main.cpp \
     $(SRCDIR)/parallelSort.cpp \
     $(SRCDIR)/dataGen.cpp \
     $(SRCDIR)/stlSort.cpp

OBJS=$(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

CXXFLAGS+=-O3 -std=c++0x #-Wall -Wextra
LDFLAGS+=-lpthread -lmpi -lmpi_cxx -Llib -lsort

.PHONY: jobs

# all should come first in the file, so it is the default target!
all : parallelSort

run : parallelSort
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) $(MPIRUN) -np 4 parallelSort -s 100000000 -d norm -p 5

parallelSort: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

jobs: parallelSort
	cd jobs && ./generate_job.sh 1
	cd jobs && ./generate_job.sh 2
	cd jobs && ./generate_job.sh 4
	cd jobs && ./generate_job.sh 8
	cd jobs && ./generate_job.sh 16
	cd jobs && ./generate_job.sh 32
	cd jobs && ./generate_job.sh 64
	cd jobs && ./generate_job.sh 128

$(OBJS): | $(OBJDIR)
$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJDIR) parallelSort $(TOOLS) jobs/$(USER)_*.job

