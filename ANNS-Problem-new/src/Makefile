HEADERS = $(wildcard *.h)
SRC = $(wildcard *.cpp)
EXE = $(SRC:.cpp=)

CXXFLAGS = -O3 -fopenmp

all: $(EXE)

$(EXE): %: %.cpp $(HEADERS)
	g++ -o $@ $<

.PHONY: clean
clean:
	rm -f $(EXE)