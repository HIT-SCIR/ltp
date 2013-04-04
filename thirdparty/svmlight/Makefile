#
# makefile for svm_light 
#
# Thorsten Joachims, 2002
#

#Use the following to compile under unix or cygwin
CC = gcc
LD = gcc

#Uncomment the following line to make CYGWIN produce stand-alone Windows executables
#SFLAGS= -mno-cygwin

CFLAGS=  $(SFLAGS) -O3                     # release C-Compiler flags
LFLAGS=  $(SFLAGS) -O3                     # release linker flags
#CFLAGS= $(SFLAGS) -pg -Wall -pedantic      # debugging C-Compiler flags
#LFLAGS= $(SFLAGS) -pg                      # debugging linker flags
LIBS=-L. -lm                               # used libraries

all: svm_learn_hideo svm_classify

tidy: 
	rm -f *.o 
	rm -f pr_loqo/*.o

clean:	tidy
	rm -f svm_learn
	rm -f svm_classify
	rm -f libsvmlight.so

help:   info

info:
	@echo
	@echo "make for SVM-light               Thorsten Joachims, 1998"
	@echo
	@echo "Thanks to Ralf Herbrich for the initial version."
	@echo 
	@echo "USAGE: make [svm_learn | svm_learn_loqo | svm_learn_hideo | "
	@echo "             libsvmlight_hideo | libsvmlight_loqo | "
	@echo "             svm_classify | all | clean | tidy]"
	@echo 
	@echo "    svm_learn           builds the learning module (prefers HIDEO)"
	@echo "    svm_learn_hideo     builds the learning module using HIDEO optimizer"
	@echo "    svm_learn_loqo      builds the learning module using PR_LOQO optimizer"
	@echo "    svm_classify        builds the classfication module"
	@echo "    libsvmlight_hideo   builds shared object library that can be linked into"
	@echo "                        other code using HIDEO"
	@echo "    libsvmlight_loqo    builds shared object library that can be linked into"
	@echo "                        other code using PR_LOQO"
	@echo "    all (default)       builds svm_learn + svm_classify"
	@echo "    clean               removes .o and target files"
	@echo "    tidy                removes .o files"
	@echo

# Create executables svm_learn and svm_classify

svm_learn_hideo: svm_learn_main.o svm_learn.o svm_common.o svm_hideo.o 
	$(LD) $(LFLAGS) svm_learn_main.o svm_learn.o svm_common.o svm_hideo.o -o svm_learn $(LIBS)

#svm_learn_loqo: svm_learn_main.o svm_learn.o svm_common.o svm_loqo.o loqo
#	$(LD) $(LFLAGS) svm_learn_main.o svm_learn.o svm_common.o svm_loqo.o pr_loqo/pr_loqo.o -o svm_learn $(LIBS)

svm_classify: svm_classify.o svm_common.o 
	$(LD) $(LFLAGS) svm_classify.o svm_common.o -o svm_classify $(LIBS)


# Create library libsvmlight.so, so that external code can get access to the
# learning and classification functions of svm-light by linking this library.

svm_learn_hideo_noexe: svm_learn_main.o svm_learn.o svm_common.o svm_hideo.o 

libsvmlight_hideo: svm_learn_main.o svm_learn.o svm_common.o svm_hideo.o 
	$(LD) -shared svm_learn.o svm_common.o svm_hideo.o -o libsvmlight.so

#svm_learn_loqo_noexe: svm_learn_main.o svm_learn.o svm_common.o svm_loqo.o loqo

#libsvmlight_loqo: svm_learn_main.o svm_learn.o svm_common.o svm_loqo.o 
#	$(LD) -shared svm_learn.o svm_common.o svm_loqo.o  pr_loqo/pr_loqo.o -o libsvmlight.so

# Compile components

svm_hideo.o: svm_hideo.c
	$(CC) -c $(CFLAGS) svm_hideo.c -o svm_hideo.o 

#svm_loqo.o: svm_loqo.c 
#	$(CC) -c $(CFLAGS) svm_loqo.c -o svm_loqo.o 

svm_common.o: svm_common.c svm_common.h kernel.h
	$(CC) -c $(CFLAGS) svm_common.c -o svm_common.o 

svm_learn.o: svm_learn.c svm_common.h
	$(CC) -c $(CFLAGS) svm_learn.c -o svm_learn.o 

svm_learn_main.o: svm_learn_main.c svm_learn.h svm_common.h
	$(CC) -c $(CFLAGS) svm_learn_main.c -o svm_learn_main.o 

svm_classify.o: svm_classify.c svm_common.h kernel.h
	$(CC) -c $(CFLAGS) svm_classify.c -o svm_classify.o

#loqo: pr_loqo/pr_loqo.o

#pr_loqo/pr_loqo.o: pr_loqo/pr_loqo.c
#	$(CC) -c $(CFLAGS) pr_loqo/pr_loqo.c -o pr_loqo/pr_loqo.o

