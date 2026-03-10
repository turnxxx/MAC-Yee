Machine = LSSC
Machine = Local
DEBUG = 0

ifeq ($(Machine),Local)
LIBPATH = /home/eureka/Libraries
ifeq ($(DEBUG),1)
DIRNAME = mfem-master-make-debug
else
DIRNAME = mfem-master-make
endif
endif

ifeq ($(Machine),LSSC)
LIBPATH = /share/home/xiruijie/Libraries
ifeq ($(DEBUG),1)
DIRNAME = mfem-4.7-mvapich2-gcc-debug
else
DIRNAME = mfem-4.7-mvapich2-gcc
endif
endif

MFEM_INSTALL_DIR = $(LIBPATH)/$(DIRNAME)
CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk
ADDITIONAL_FLAGS = -Wno-unused-variable

include $(CONFIG_MK)

.SUFFIXES:
# .SUFFIXES: .o .cpp .mk
.PHONY: all clean cleanall

%.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) $(ADDITIONAL_FLAGS) -c $<
	
OBJECTS = Hall_MHD_dual.o \
			tools.o \
			Integrators.o \
			testcase_temporal.o \
			testcase_spatial.o \
			testcase_spatial2.o \
			testcase_spatial_test.o \
			testcase_conservation.o \
			testcase_island.o \
			testcase_island_guo.o \
			testcase_adaptive.o \
			testcase_Whistler.o \
			testcase_MHDvortex.o \
			testcase_loop.o \
			testcase_OrszagTang.o \
			testcase_OrszagTang_Kraus.o \
			testcase_OrszagTang_3d.o \
			testcase_TGVortex.o \
			testcase_Hopf.o \
			testcase_friction.o \
			testcase_island_origin.o \
			testcase_island_Adler.o \
			ErrorEstimator.o \
			refiner.o \
			MHD_solver.o \
			preconditioners.o
   
Hall_MHD_dual: $(OBJECTS)
	$(MFEM_CXX) $(MFEM_FLAGS) $^ -o $@ $(MFEM_LIBS) $(ADDITIONAL_FLAGS)
	
test_point: test_point.o tools.o
	$(MFEM_CXX) $(MFEM_FLAGS) $^ -o $@ $(MFEM_LIBS)
	
test-weakcurl: test-weakcurl.o
	$(MFEM_CXX) $(MFEM_FLAGS) $^ -o $@ $(MFEM_LIBS)
	
default: Hall_MHD_dual

clean:
	@rm -f Hall_MHD_dual *.o test_point test-weakcurl

