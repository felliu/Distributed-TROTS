MKL_LINK_FLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
MKL_CXX_FLAGS = -m64 -I"${MKLROOT}/include"

CXX_FLAGS = -O3 -g -std=c++17

trots_test: main.o trots.o TROTSEntry.o Makefile
	g++ main.o trots.o TROTSEntry.o ${MKL_LINK_FLAGS} -lmatio -O3 -o trots_test

main.o: main.cpp trots.h Makefile
	g++ ${CXX_FLAGS} -c main.cpp -o main.o

trots.o: trots.cpp trots.h TROTSEntry.h SparseMat.h util.h Makefile
	g++ ${CXX_FLAGS} ${MKL_CXX_FLAGS} -c trots.cpp -o trots.o

TROTSEntry.o: TROTSEntry.cpp TROTSEntry.h SparseMat.h util.h Makefile
	g++ ${CXX_FLAGS} -c TROTSEntry.cpp -o TROTSEntry.o

