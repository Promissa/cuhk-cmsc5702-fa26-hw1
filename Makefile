CXX = g++
MPICXX ?= mpic++
CXXFLAGS ?= -O3
OMPFLAGS ?= -fopenmp

PROGRAMS = heat_seq heat_omp heat_mpi
LIVE_PROGRAMS = heat_seq_live heat_omp_live heat_mpi_live
BINARIES = $(addprefix build/,$(PROGRAMS) $(LIVE_PROGRAMS))

.PHONY: all live clean submission $(PROGRAMS) $(LIVE_PROGRAMS)
all: $(addprefix build/,$(PROGRAMS))
live: $(addprefix build/,$(LIVE_PROGRAMS))

$(PROGRAMS) $(LIVE_PROGRAMS): %: build/%

build:
	mkdir -p $@

build/heat_seq: src/heat_seq.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/heat_omp: src/heat_omp.cpp | build
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

build/heat_mpi: src/heat_mpi.cpp | build
	$(MPICXX) $(CXXFLAGS) $< -o $@

build/heat_seq_live: src/heat_seq.cpp | build
	$(CXX) $(CXXFLAGS) -DLIVE $< -o $@

build/heat_omp_live: src/heat_omp.cpp | build
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DLIVE $< -o $@

build/heat_mpi_live: src/heat_mpi.cpp | build
	$(MPICXX) $(CXXFLAGS) -DLIVE $< -o $@

submission:
	mkdir -p submission
	cp src/heat_omp.cpp src/heat_mpi.cpp submission/
	cp data/timings/seq_times.csv data/timings/omp_times.csv data/timings/mpi_times.csv submission/
	cp reports/report.ipynb submission/

clean:
	rm -f $(BINARIES)
