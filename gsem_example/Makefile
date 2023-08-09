source = $(wildcard *.cpp)
object = $(patsubst %.cpp, %.o, $(source))
path_afepack = /home/hfzhan/Packages/afepack-0.1/library/include
path_dealii = /usr/local/include/deal.II
path_tsem = ./include
path_lapacke = /home/hfzhan/Packages/lapack-3.11/LAPACKE/include
path_lapacke_so = /lib/x86_64-linux-gnu/liblapack.so.3


all : main

%.o : %.cpp
	mpicxx -c -o $@ $< -I$(path_afepack) -I/usr/local/include -I$(path_dealii) -D__SERIALIZATION__ -DMULTITHREAD -pthread -fno-delete-null-pointer-checks -O2 -I$(path_tsem) -I$(path_lapacke)

main : $(object)
	mpicxx -g -o $@ $(object) -L/usr/local/lib -ldeal_II -ltbb -ldl -lm -pthread -lAFEPack -O2 -fno-delete-null-pointer-checks -fPIC -D__SERIALIZATION__ -DMULTITHREAD -pthread -I$(path_afepack) -I$(path_tsem) $(path_lapacke_so)

clean :
	-rm -rf $(object)
	-rm -rf main

.PHONY : default clean
