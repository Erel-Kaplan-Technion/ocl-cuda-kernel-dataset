# Makefile to build shrUtils.cpp into libshrutil_x86_64.a for x86_64 Linux

CXX = g++
AR = ar
CXXFLAGS = -O3 -fPIC -m64 -I../inc -I../../OpenCL/common/inc -I../inc -DUNIX
SRCS = shrUtils.cpp cmd_arg_reader.cpp
OBJS = $(SRCS:.cpp=.o)
LIBDIR = ../lib/x64
LIBNAME = libshrutil_x86_64.a

all: $(LIBDIR)/$(LIBNAME)

$(LIBDIR)/$(LIBNAME): $(OBJS) | $(LIBDIR)
	$(AR) rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIBDIR):
	mkdir -p $(LIBDIR)

clean:
	rm -f *.o $(LIBDIR)/$(LIBNAME)

