# - If the library source code has changed at all since the last update,
# then increment revision.
# - If any interfaces have been added, removed, or changed since the last
# update, increment current, and set revision to 0.
# - If any interfaces have been added since the last public release, then
# increment age.
# - If any interfaces have been removed since the last public release, then
# set age to 0.

m4_define([lib_current],0)
m4_define([lib_revision],4)
m4_define([lib_age],0)

# Setup autoconf
AC_INIT(eegdev,[1.0],[nicolas.bourdaud@gmail.com])
AC_CONFIG_SRCDIR([src/core/eegdev.h])
AC_CONFIG_LIBOBJ_DIR([lib])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADERS([config/config.h])

# Setup automake
AM_INIT_AUTOMAKE([no-dist-gzip dist-bzip2 color-tests subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])


AC_SET_HOSTSYSTEM
AC_CANONICAL_BUILD

# Check for programs
AC_PROG_CC_C99

AM_PROG_CC_C_O
LT_INIT([win32-dll disable-static])
LT_SYS_MODULE_EXT
AC_SUBST(LIBTOOL_DEPS)
AC_CHECK_PROG(HAS_FLEX,flex,yes,no)
AC_CHECK_PROG(HAS_BISON,bison,yes,no)


AC_C_BIGENDIAN
AC_DEF_API_EXPORT_ATTRS

# MMLibrary is needed in core and modules
AC_SEARCH_LIBS([mm_open], [mmlib], [],
               AC_MSG_ERROR([The mmlib library has not been found]))

# Test whether the core library should be build
save_LIBS=$LIBS
AC_ARG_ENABLE([corelib-build], AC_HELP_STRING([--enable-corelib-build],
	                [Build the core library. Disabling it may be useful
			when only some plugins must be rebuilt @<:@default=yes@:>@]),
	    [], [enable_corelib_build=yes])
AS_IF([test "x$enable_corelib_build" = xno], 
      [AC_SEARCH_LIBS([egd_open], [eegdev], [], 
        [AC_MSG_FAILURE([eegdev library required since it is not built])])],
      [AS_IF([test "x$HAS_FLEX" = xyes && test "x$HAS_BISON" = xyes],
             [enable_corelib_build=yes],
	     [AC_MSG_ERROR([flex and bison needed to build the core lib])])])
AM_CONDITIONAL([BUILD_CORELIB], [test "x$enable_corelib_build" = xyes])
AC_SUBST([CORE_EEGDEV_LIBS], "$LIBS")
LIBS=$save_LIBS

# Test for sphinx doc with linuxdoc
AC_ARG_ENABLE([sphinxdoc], AC_HELP_STRING([--enable-sphinxdoc], [Build sphinx documention. @<:@default=detect@:>@]),
	    [], [enable_sphinxdoc=detect])
AM_PATH_PYTHON([3])
MM_PYTHON_MODULE([sphinx])
MM_PYTHON_MODULE([sphinx_rtd_theme])
MM_PYTHON_MODULE([linuxdoc])

AC_CHECK_PROG(HAVE_SPHINXBUILD, [sphinx-build], [$HAVE_PYMOD_LINUXDOC], [no])
AS_CASE([$enable_sphinxdoc],
        [yes], [AS_IF([test "x$HAVE_SPHINXBUILD" != xyes],
                      [AC_MSG_ERROR(Cannot find sphinx or its plugin linuxdoc)])],
        [detect], [AS_IF([test "x$HAVE_SPHINXBUILD" = xyes],
                         [enable_sphinxdoc=yes], [enable_sphinxdoc=no])])
AM_CONDITIONAL(BUILD_SPHINXDOC, [test "x$enable_sphinxdoc" = xyes])

# EEGfile support
save_LIBS=$LIBS
AC_ARG_WITH([xdf], AC_HELP_STRING([--with-xdf],
	               [Support for EEGfile backend @<:@default=check@:>@]),
	    [], [with_xdf=check])
AS_IF([test "x$with_xdf" != xno], [
     AC_SEARCH_LIBS([xdf_open], [xdffileio], [xdf_support=yes],
        [xdf_support=no; if test "x$with_xdf" != xcheck; then
            AC_MSG_FAILURE([xdffileio library required for EEGfile support])
         fi])],
     [xdf_support=no])
AS_IF([test "x$with_xdf" != xno], [
     AC_SEARCH_LIBS([tre_regcomp], [tre], [],
        [xdf_support=no; if test "x$with_xdf" != xcheck; then
            AC_MSG_FAILURE([tre library required for EEGfile support])
         fi])],
     [xdf_support=no])
AM_CONDITIONAL([XDF_SUPPORT], [test "x$xdf_support" = "xyes"])
AS_IF([test "x$xdf_support" = "xyes"], optional_examples=recinxdf$EXEEXT)
AC_SUBST([XDF_LIBS], "$LIBS")
LIBS=$save_LIBS



# Biosemi support
save_LIBS=$LIBS
AC_ARG_WITH([act2], AC_HELP_STRING([--with-act2],
	               [Support for Biosemi backend @<:@default=check@:>@]),
	    [], [with_act2=check])
AS_IF([test "x$with_act2" != xno],
    [PKG_CHECK_MODULES(LIBUSB, [libusb-1.0], [act2_support=yes],
        [act2_support=no; if test "x$with_act2" != xcheck; then
             AC_MSG_FAILURE([libusb-1.0 library required for Biosemi support])
         fi])],
    [act2_support=no])
LIBS="$LIBUSB_LIBS $LIBS"
AM_CONDITIONAL([ACT2_SUPPORT], [test "x$act2_support" = xyes])
AC_SUBST([ACT2_LIBS], "$LIBS")
AC_SUBST([CPPFLAGS], "$LIBUSB_CFLAGS $CPPFLAGS")
LIBS=$save_LIBS



# gTec support
save_LIBS=$LIBS
AC_ARG_WITH([gtec], AC_HELP_STRING([--with-gtec],
		         [Support for gTec backend @<:@default=check@:>@]),
	    [], [with_gtec=check])
gtec_support=no
AS_IF([test "x$with_gtec" != xno && test "x$with_gtec" != xmodule],
      [AC_SEARCH_LIBS([GT_OpenDevice], [gUSBampAPI gusbampapi],
                                       [gtec_support=yes])])
AS_IF([test "x$with_gtec" = xyes && test "x$gtec_support" = xno],
       AC_MSG_FAILURE([gUSBampAPI library required for gTec support]))
AC_DEFINE([DLOPEN_GUSBAMP], [0], [Define to 1 if gusbamp must dlopened])
AM_CONDITIONAL([DLOPEN_GUSBAMP], [test "x$with_gtec" = xmodule])
AS_IF([test "x$with_gtec" = xmodule],
      [dlopen_gusbamp=yes; gtec_support=yes;
       AC_DEFINE([DLOPEN_GUSBAMP], [1])],
      [dlopen_gusbamp=no])
AS_IF([test "x$with_gtec" != xno && test "x$gtec_support" = xyes],
     [AC_SEARCH_LIBS([pthread_create], [pthread], [gtec_support=yes],
            [AC_MSG_FAILURE([pthread library required for gTec support]);
			gtec_support=no]
         )])
AC_SUBST([DLOPEN_GUSBAMP], "$dlopen_gusbamp")
AM_CONDITIONAL([GTEC_SUPPORT], [test "x$gtec_support" = xyes])
AC_SUBST([GTEC_LIBS], "$LIBS")
LIBS=$save_LIBS



# Neurosky support
save_LIBS=$LIBS
AC_ARG_WITH([neurosky], AC_HELP_STRING([--with-neurosky],
	                [Support for Neurosky backend @<:@default=no@:>@]),
	    [], [with_neurosky=no])
AS_IF([test "x$with_neurosky" != xno],
      AC_SEARCH_LIBS([str2ba], [bluetooth], [neurosky_support=yes],
        [neurosky_support=no; if test "x$with_neurosky" != xcheck; then
             AC_MSG_FAILURE([bluetooth library required for neurosky support])
         fi]), [neurosky_support=no])
AM_CONDITIONAL([NSKY_SUPPORT], [test "x$neurosky_support" = xyes])
AC_SUBST([NSKY_LIBS], "$LIBS")
LIBS=$save_LIBS



# Tobi interface A support
save_LIBS=$LIBS
AC_ARG_WITH([tia], AC_HELP_STRING([--with-tia],
	      [Support for Tobi interface A backend @<:@default=check@:>@]),
	    [], [with_tia=check])
AS_IF([test "x$with_tia" != xno],
    [AC_SEARCH_LIBS([XML_Parse], [expat], [tia_support=yes],
        [tia_support=no; if test "x$with_tia" != xcheck; then
             AC_MSG_FAILURE([the expat library is required for Tobi interface A support])
         fi])], [tia_support=no])
AM_CONDITIONAL([TIA_SUPPORT], [test "x$tia_support" = xyes])
AC_SUBST([TIA_LIBS], "$LIBS")
LIBS=$save_LIBS


AC_SUBST([optional_examples])
AC_SUBST([CURRENT],[lib_current])
AC_SUBST([REVISION],[lib_revision])
AC_SUBST([AGE],[lib_age])

# Add ifndef in config.h to fix pthread-win32 header mess
AH_TOP([#ifndef CONFIG_H
#define CONFIG_H])
AH_BOTTOM([#endif /*CONFIG_H*/])

AC_CONFIG_FILES([Makefile
                 src/core/Makefile src/core/eegdev.pc
		 src/plugins/Makefile
                 tests/Makefile tests/fakelibs/Makefile
		 doc/Makefile doc/egd_open.3
		 doc/examples/Makefile
		 doc/examples/library-usage/Makefile])
AC_OUTPUT

echo "

Configuration summary
=====================
    Core library build : $enable_corelib_build
    --------------------------
    EEG file support : $xdf_support
    Biosemi support  : $act2_support
    gTec support     : $gtec_support (loaded at runtime: $dlopen_gusbamp)
    Neurosky support : $neurosky_support
    TobiIA support   : $tia_support
"

