pkglib_LTLIBRARIES =

AM_CPPFLAGS=$(CPPFLAGS) -I$(top_srcdir)/src/core \
            -I$(top_builddir)/lib -I$(top_srcdir)/lib
AM_LDFLAGS=-module -avoid-version -no-undefined

pkgsysconfdir = $(sysconfdir)/$(PACKAGE_NAME)
pkgsysconf_DATA =

if ACT2_SUPPORT
pkglib_LTLIBRARIES += biosemi.la
pkgsysconf_DATA += biosemi.conf
endif
biosemi_la_SOURCES = biosemi.c
biosemi_la_LIBADD = $(ACT2_LIBS) $(LIB_MMLIB)

if XDF_SUPPORT
pkglib_LTLIBRARIES += datafile.la
endif
datafile_la_SOURCES = fileout.c
datafile_la_LIBADD = $(XDF_LIBS) $(LIB_TRE) $(LIB_MMLIB)

if NSKY_SUPPORT
pkglib_LTLIBRARIES += neurosky.la
endif
neurosky_la_SOURCES = neurosky.c
neurosky_la_LIBADD = $(NSKY_LIBS)

if GTEC_SUPPORT
pkglib_LTLIBRARIES += gtec.la
endif
gtec_la_SOURCES = gtec.c
gtec_la_LIBADD = $(PTHREAD) $(LIB_MMLIB)
if DLOPEN_GUSBAMP
gtec_la_SOURCES += gusbamp-types.h gusbamp-loader.h gusbamp-loader.c
else
gtec_la_LIBADD += $(GTEC_LIBS)
endif


if TIA_SUPPORT
pkglib_LTLIBRARIES += tobiia.la
endif
tobiia_la_SOURCES = tobiia.c
tobiia_la_LIBADD = $(TIA_LIBS) $(LIB_MMLIB)
