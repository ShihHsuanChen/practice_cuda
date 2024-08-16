# Main variables
MAIN := main.cu
BUILDDIR := build
TARGET := bin/main

# Other variables
TARGETDIR = `dirname $(TARGET)`

# Compilers
NVCC := nvcc #/usr/local/cuda/bin/nvcc

# Flags
NVCCFLAGS := 

# Target rules
all: build

build: $(TARGET)

run: build
	./$(TARGET)

clean:
	rm -rf $(TARGET)

clobber:
	rm -rf $(TARGETDIR)

$(TARGET): 
	@mkdir -p $(TARGETDIR);
	$(NVCC) $(NVCCFLAGS) -o $@ $(MAIN)
	@chmod a+x $(TARGET)
