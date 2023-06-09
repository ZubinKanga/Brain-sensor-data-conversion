subdir('examples')

docs_state = 'disabled'
sphinxbuild = find_program('sphinx-build', required : get_option('docs'))
if sphinxbuild.found() and not get_option('docs').disabled()
    python3 = import('python').find_installation('python3', required : true)
    check_linuxdoc = run_command(python3, '-c', '"import linuxdoc"')

    if check_linuxdoc.returncode() != 0 and get_option('docs').enabled()
        error('python3 module "linuxdoc" is required to build documentation')
    elif check_linuxdoc.returncode() == 0
        docs_state = 'enabled'
    endif
endif

if docs_state == 'enabled'

    doc_sources = files(
        'index.rst',
        'eegdev.rst',
    )

    sphinxbuild_wrapper = files('sphinx-build-wrapper.sh')
    gen_man_pages = custom_target('man3',
            output : 'man3',
            command : [
                'sh',
                sphinxbuild_wrapper,
                sphinxbuild,
                meson.source_root(),
                'kernel-doc-man',
                meson.source_root() / 'doc',
                'doc/man3',
            ],
            build_by_default : true,
            depend_files : [eegdev_sources, doc_sources],
            install : true,
            install_dir : get_option('mandir'),
    )

    custom_target('html',
            output : 'html',
            command : [
                'sh',
                sphinxbuild_wrapper,
                sphinxbuild,
                meson.source_root(),
                'html',
                meson.source_root() / 'doc',
                'doc/html',
            ],
            build_by_default : true,
            depend_files : [eegdev_sources, doc_sources],
            depends : gen_man_pages,  # re-use .doctree from man
            install : true,
            install_dir : get_option('datadir') / 'doc/eegdev',
    )

endif


if get_option('corelib-build').enabled()
    man_pages = files(
        'eegdev-open-options.5',
    )
    install_man(man_pages)
endif

if xdf_state == 'enabled'
    install_man('eegdev-datafile.5')
endif
if neurosky_state == 'enabled'
    install_man('eegdev-neurosky.5')
endif
if biosemi_state == 'enabled'
    install_man('eegdev-biosemi.5')
endif
if gtec_state == 'enabled'
    install_man('eegdev-gtec.5')
endif
if tobiia_state == 'enabled'
    install_man('eegdev-tobiia.5')
endif
