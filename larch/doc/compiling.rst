

.. _compiling:

Compiling from Source
=====================

It is highly recommended that you not compile Larch from the source code, but instead
just use the pre-built packages available through conda.

If you do attempt to compile from source and have some trouble, here are some tips:


MacOS
-----

-   You will find it easier to use *clang* and *openmp* from conda, but you probably
    also need the plain XCode installed (get it from the Mac App Store).

    .. code-block:: console

        xcode-select --install
        conda install clangdev llvmdev openmp -c conda-forge

-   If you encounter `fatal error: 'stdio.h' file not found` when compiling,
    you may need to update your SDK headers.  `See a discussion here
    <https://stackoverflow.com/questions/52509602/cant-compile-c-program-on-a-mac-after-upgrade-to-mojave>`_
    or skip to the solution:

    .. code-block:: console

        open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg


Windows
-------

-   Don't.

-   The Windows build for Larch is generated automatically after successful testing on
    *Appveyor*.  If you want to build Larch for Windows yourself, the first step is to
    look deep inside your soul and ask yourself if it's really worth giving up that much
    of your life to do so.  If after doing that you still want to proceed, you can try
    to follow along with the build instructions from *appveyor.yml* in the Larch github
    repository.




