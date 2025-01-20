EXECS=pp35mpi
COMPILER?=mpic++

all: ${EXECS}

pp35mpi: pp35mpi.cpp
	${COMPILER} -o $@ -std=c++11 -g -lgomp pp35mpi.cpp

clean:
	rm ${EXECS}
