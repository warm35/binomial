# CUDA SDK 10 Linux Version 1.10.1112.1225
PROJECTS := $(shell find projects -name Makefile)

%.ph_build : lib/libcutil.so lib/libparamgl.so
	make -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	make -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	make -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building all"

lib/libcutil.so:
	@make -C common

lib/libparamgl.so:
	@make -C common -f Makefile_paramgl

tidy:
	@find | egrep "#" | xargs rm -f
	@find | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))
	@make -C common clean

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
	@make -C common clobber
