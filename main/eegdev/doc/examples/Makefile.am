# Verify the examples compile and execute
if BUILD_CORELIB
check_PROGRAMS = eegdev_acq
check_LTLIBRARIES = saw.la
TESTS = testexamples.sh

if XDF_SUPPORT
check_PROGRAMS += recinxdf
endif

# install examples files
exampledir = $(docdir)/examples
nobase_dist_example_DATA = README			\
                           $(eegdev_acq_SOURCES)	\
                           $(recinxdf_SOURCES)		\
                           $(saw_la_SOURCES)		\
			   plugin/Makefile		\
                           plugin/README
endif


TESTS_ENVIRONMENT = EEGDEV_PLUGINS_DIR=.libs EXEEXT=$(EXEEXT)\
                    PATH=$(builddir):$(PATH)
EXTRA_DIST=testexamples.sh

AM_CPPFLAGS = -I$(top_srcdir)/src/core -I$(top_srcdir)/lib


eegdev_acq_SOURCES = library-usage/eegdev_acq.c
eegdev_acq_LDADD = $(top_builddir)/src/core/libeegdev.la


recinxdf_SOURCES = library-usage/acquisition.c library-usage/acquisition.h\
                   library-usage/recinxdf.c
recinxdf_LDADD = $(top_builddir)/src/core/libeegdev.la -lxdffileio


saw_la_SOURCES = plugin/saw.c
saw_la_LDFLAGS=-module -avoid-version -no-undefined -rpath /nowhere
saw_la_LIBADD = $(LIB_MMLIB)


