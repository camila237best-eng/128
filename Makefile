# Makefile - puzzle.cu (GPU + CPU) et docta.cpp (CPU seul)
#
#   make              GPU si nvcc est trouvé (-> ./puzzle), sinon version CPU seule (-> ./puzzle_cpu)
#   make gpu          force la version GPU + CPU      (nvcc)  -> ./puzzle
#   make cpu          force la version CPU seule      (g++)   -> ./puzzle_cpu
#   make docta        version CPU seule de docta.cpp  (g++)   -> ./docta
#   make clean        supprime les exécutables
#
# Options :
#   make SM=75        architecture GPU : 75 = T4 (Kaggle/Colab), 60 = P100, 80 = A100,
#                     89 = L4 / RTX 4090 / RTX 4080, 90 = H100, 120 = RTX 5090 / RTX 5080 (Blackwell)
#                     (par défaut : -arch=native, qui détecte la carte présente automatiquement ;
#                      demande CUDA >= 11.6, et CUDA >= 12.8 pour une carte Blackwell comme la RTX 5090 ;
#                      si nvcc refuse l'architecture détectée, indique SM=... ou mets à jour CUDA)
#   make NVCC=/usr/local/cuda/bin/nvcc      chemin de nvcc
#   make CXX=clang++                        autre compilateur C++
#   make EXTRA=-mno-avx                     options supplémentaires (ex. Windows/MinGW qui plante : -mno-avx)

NVCC  ?= nvcc
CXX   ?= g++
SM    ?=
EXTRA ?=

ifeq ($(SM),)
GENCODE := -arch=native
else
GENCODE := -gencode arch=compute_$(SM),code=sm_$(SM)
endif

HAS_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes)

.PHONY: all gpu cpu docta clean help

ifeq ($(HAS_NVCC),yes)
all: puzzle
else
all: puzzle_cpu
	@echo "nvcc introuvable : version CPU seule compilée (puzzle_cpu). Pour le GPU : make gpu NVCC=/chemin/vers/nvcc"
endif

gpu: puzzle
cpu: puzzle_cpu

puzzle: puzzle.cu
	$(NVCC) -O3 -std=c++17 $(GENCODE) -Xcompiler -O3,-march=native,-pthread -lpthread $(EXTRA) -o $@ $<

puzzle_cpu: puzzle.cu
	$(CXX) -x c++ -O3 -march=native -std=c++17 -pthread $(EXTRA) -o $@ $<

docta: docta.cpp
	$(CXX) -O3 -march=native -std=c++17 -pthread $(EXTRA) -o $@ $<

clean:
	rm -f puzzle puzzle_cpu docta

help:
	@sed -n '1,/^NVCC/p' Makefile | sed '$$d' | sed 's/^# \{0,1\}//'
