AM_CPPFLAGS = -DPKGLIBDIR='"$(pkglibdir)"' \
	      -DPKGSYSCONFDIR='"$(sysconfdir)/$(PACKAGE_NAME)"'
AM_CFLAGS = $(CFLAG_VISIBILITY)

lib_LTLIBRARIES =
BUILT_SOURCES =
CLEANFILES =

defexecdir = $(bindir)
defexec_DATA =
DISTCLEANFILES = $(defexec_DATA)

EXTRA_DIST = configuration.l configuration.y
GENERATED = configuration.lex.c configuration.lex.h \
            configuration.tab.c configuration.tab.h

%.lex.c %.lex.h: %.l
	$(AM_V_GEN)flex $^

%.tab.c %.tab.h: %.y
	$(AM_V_GEN)bison $^


libeegdev_la_SOURCES = eegdev.h eegdev-pluginapi.h core.c	\
		       coreinternals.h typecast.c device-helper.c	\
		       opendev.c sensortypes.c \
		       configuration.h confparser.h
nodist_libeegdev_la_SOURCES = $(GENERATED)

libeegdev_la_LDFLAGS = $(AM_LDFLAGS) -no-undefined			\
                       -version-info $(CURRENT):$(REVISION):$(AGE)
libeegdev_la_LIBADD = $(LIB_MMLIB)

pkgconfigdir = $(libdir)/pkgconfig

if BUILD_CORELIB
BUILT_SOURCES += $(GENERATED)
CLEANFILES += $(GENERATED)
lib_LTLIBRARIES += libeegdev.la
include_HEADERS = eegdev.h eegdev-pluginapi.h
pkgconfig_DATA = eegdev.pc
endif
