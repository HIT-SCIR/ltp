Armadillo C++ Linear Algebra Library
http://arma.sourceforge.net



Contents
========

 1: Introduction
 2: Citation Details
 3: Requirements
 
 4: Linux and Mac OS X: Installation
 5: Linux and Mac OS X: Compiling & Linking
 
 6: Windows: Installation
 7: Windows: Compiling & Linking
 
 8: Support for OpenBLAS, Intel MKL and AMD ACML
 9: Support for ATLAS
 
10: Documentation / API Reference Manual
11: MEX Interface to Octave
12: Bug Reports and Frequently Asked Questions

13: License
14: Developers and Contributors
15: Related Software



1: Introduction
===============

Armadillo is a C++ linear algebra library (matrix maths)
aiming towards a good balance between speed and ease of use.
The syntax is deliberately similar to Matlab.

The library provides efficient classes for vectors, matrices and cubes,
as well as many functions which operate on the classes
(eg. contiguous and non-contiguous submatrix views).

Integer, floating point and complex numbers are supported,
as well as a subset of trigonometric and statistics functions.
Various matrix decompositions are provided through optional
integration with LAPACK or high-performance LAPACK-compatible
libraries (such as Intel MKL or AMD ACML).

A delayed evaluation approach is automatically employed (at compile time)
to combine several operations into one and reduce (or eliminate)
the need for temporaries. This is accomplished through recursive
templates and template meta-programming.

This library is useful for conversion of research code into
production environments, or if C++ has been decided as the
language of choice, due to speed and/or integration capabilities.

The library is open-source software, and is distributed under a license
that is useful in both open-source and commercial/proprietary contexts.

Armadillo is primarily developed at NICTA (Australia),
with contributions from around the world.  More information
about NICTA can be obtained from http://nicta.com.au

Main developers:
  Conrad Sanderson - http://conradsanderson.id.au
  Ryan Curtin      - http://ratml.org



2: Citation Details
===================

Please cite the following tech report if you use Armadillo in your
research and/or software. Citations are useful for the continued
development and maintenance of the library.

  Conrad Sanderson.
  Armadillo: An Open Source C++ Linear Algebra Library for
  Fast Prototyping and Computationally Intensive Experiments.
  Technical Report, NICTA, 2010.



3: Requirements
===============

Armadillo makes extensive use of template meta-programming, recursive templates
and template based function overloading.  As such, C++ compilers which do not
fully implement the C++ standard may not work correctly.

The functionality of Armadillo is partly dependent on other libraries:
LAPACK, BLAS, ARPACK and SuperLU.  The LAPACK and BLAS libraries are
used for dense matrices, while the ARPACK and SuperLU libraries are
used for sparse matrices.  Armadillo can work without these libraries,
but its functionality will be reduced. In particular, basic functionality
will be available (eg. matrix addition and multiplication), but things
like eigen decomposition or matrix inversion will not be.
Matrix multiplication (mainly for big matrices) may not be as fast.

As Armadillo is a template library, we recommended that optimisation
is enabled during compilation of programs that use Armadillo.
For example, for GCC and Clang compilers use -O2 or -O3



4: Linux and Mac OS X: Installation
===================================

You can install Armadillo on your system using the procedure detailed below,
or use Armadillo without installation (detailed in section 5).

Installation procedure:

* Step 1:
  If CMake is not already be present on your system, download
  it from http://www.cmake.org
  
  On major Linux systems (such as Fedora, Ubuntu, Debian, etc),
  cmake is available as a pre-built package, though it may need
  to be explicitly installed (using a tool such as PackageKit,
  yum, rpm, apt, aptitude).
  
* Step 2:
  If you have LAPACK or BLAS, install them before installing Armadillo.
  Under Mac OS X this is not necessary.
  
  If you have ARPACK and/or SuperLU, install them before installing Armadillo.
  Caveat: only SuperLU version 4.3 can be used!
  
  On Linux systems it is recommended that the following libraries
  are present: LAPACK, BLAS, ARPACK, SuperLU and ATLAS.
  LAPACK and BLAS are the most important.  It is also necessary to
  install the corresponding development files for each library.
  For example, when installing the "lapack" package, also install
  the "lapack-devel" or "lapack-dev" package.
  
  For best performance, we recommend using the multi-threaded
  OpenBLAS library instead of standard BLAS.
  See http://xianyi.github.com/OpenBLAS/
  
* Step 3:
  Open a shell (command line), change into the directory that was
  created by unpacking the armadillo archive, and type the following
  commands:
  
  cmake .
  make 
  
  The full stop separated from "cmake" by a space is important.
  CMake will figure out what other libraries are currently installed
  and will modify Armadillo's configuration correspondingly.
  CMake will also generate a run-time armadillo library, which is a 
  wrapper for all the relevant libraries present on your system
  (eg. LAPACK, BLAS, ARPACK, SuperLU, ATLAS).
  
  If you need to re-run cmake, it's a good idea to first delete the
  "CMakeCache.txt" file (not "CMakeLists.txt").
  
  Caveat: out-of-tree builds are currently not fully supported;
  for example, creating a sub-directory called "build" and running cmake ..
  from within "build" is currently not supported.
  
* Step 4:
  If you have access to root/administrator/superuser privileges,
  first enable the privileges (eg. through "su" or "sudo")
  and then type the following command:
  
  make install
  
  If you don't have root/administrator/superuser privileges, 
  type the following command:
  
  make install DESTDIR=my_usr_dir
  
  where "my_usr_dir" is for storing C++ headers and library files.
  Make sure your C++ compiler is configured to use the "lib" and "include"
  sub-directories present within this directory.



5: Linux and Mac OS X: Compiling & Linking
==========================================

The "examples" directory contains several quick example programs
that use the Armadillo library.

In general, programs which use Armadillo are compiled along these lines:
  
  g++ example1.cpp -o example1 -O2 -larmadillo
  
If you want to use Armadillo without installation,
or you're getting linking errors, compile along these lines:
  
  g++ example1.cpp -o example1 -O2 -I /home/blah/armadillo-5.100.1/include -DARMA_DONT_USE_WRAPPER -lblas -llapack
  
The above command line assumes that you have unpacked the armadillo archive into /home/blah/
You will need to adjust this for later versions of Armadillo,
and/or if you have unpacked into a different directory.

Notes:

* To use the high speed OpenBLAS library instead of BLAS,
  replace -lblas -llapack with -lopenblas -llapack
  To get OpenBLAS, see http://xianyi.github.com/OpenBLAS/
  
* On most Linux-based systems, using -lblas -llapack should be enough;
  however, on Ubuntu and Debian you may need to add -lgfortran
  
* On Mac OS X, replace -lblas -llapack with -framework Accelerate
  
* If you have ARPACK present, also link with it by adding -larpack to the command line
  
* If you have SuperLU present, also link with it by adding -lsuperlu to the command line
  Caveat: only SuperLU version 4.3 can be used!
  

6: Windows: Installation
========================

The installation is comprised of 3 steps:

* Step 1:
  Copy the entire "include" folder to a convenient location
  and tell your compiler to use that location for header files
  (in addition to the locations it uses already).
  Alternatively, you can use the "include" folder directly.
  
* Step 2:
  Modify "include/armadillo_bits/config.hpp" to indicate which
  libraries are currently available on your system. For example,
  if you have LAPACK, BLAS (or OpenBLAS), ARPACK and SuperLU present,
  uncomment the following lines:
  
  #define ARMA_USE_LAPACK
  #define ARMA_USE_BLAS
  #define ARMA_USE_ARPACK
  #define ARMA_USE_SUPERLU
  
  If you don't need sparse matrices, don't worry about ARPACK or SuperLU.
  
* Step 3:
  Configure your compiler to link with LAPACK and BLAS
  (and optionally ARPACK and SuperLU).



7: Windows: Compiling & Linking
===============================

Within the "examples" folder, we have included an MSVC project named "example1_win64"
which can be used to compile "example1.cpp".  The project needs to be compiled as a
64 bit program: the active solution platform must be set to x64, instead of win32.

If you're getting messages such as "use of LAPACK needs to be enabled",
you will need to manually modify "include/armadillo_bits/config.hpp"
to enable the use of LAPACK.

The MSCV project was tested on 64 bit Windows 7 with Visual C++ 2012.
You may need to make adaptations for 32 bit systems, later versions of Windows
and/or the compiler.  For example, you may have to enable or disable
ARMA_BLAS_LONG and ARMA_BLAS_UNDERSCORE macros in "armadillo_bits/config.hpp".

The folder "examples/lib_win64" contains standard LAPACK and BLAS libraries compiled
for 64 bit Windows.  The compilation was done by a third party.  USE AT YOUR OWN RISK.
The compiled versions of LAPACK and BLAS were obtained from:
  http://ylzhao.blogspot.com.au/2013/10/blas-lapack-precompiled-binaries-for.html

You can find the original sources for standard BLAS and LAPACK at:
  http://www.netlib.org/blas/
  http://www.netlib.org/lapack/
  
Faster and/or alternative implementations of BLAS and LAPACK are available:
  http://xianyi.github.com/OpenBLAS/
  http://software.intel.com/en-us/intel-mkl/
  http://developer.amd.com/tools-and-sdks/cpu-development/amd-core-math-library-acml/
  http://icl.cs.utk.edu/lapack-for-windows/lapack/

The OpenBLAS, MKL and ACML libraries are generally the fastest.
See section 8 for more info on making Armadillo use these libraries.

For better performance, we recommend the following high-quality C++ compilers:
  GCC from MinGW:     http://www.mingw.org/
  GCC from CygWin:    http://www.cygwin.com/
  Intel C++ compiler: http://software.intel.com/en-us/intel-compilers/

For the GCC compiler, use version 4.2 or later.
For the Intel compiler, use version 11.0 or later.

For best results we also recommend using an operating system
that's more reliable and more suitable for heavy duty work,
such as Mac OS X, or various Linux-based systems:
  Ubuntu                    http://www.ubuntu.com/
  Debian                    http://www.debian.org/
  OpenSUSE                  http://www.opensuse.org/
  Fedora                    http://fedoraproject.org/
  Scientific Linux          http://www.scientificlinux.org/
  CentOS                    http://centos.org/
  Red Hat Enterprise Linux  http://www.redhat.com/



8: Support for OpenBLAS, Intel MKL and AMD ACML
===============================================

Armadillo can use OpenBLAS, or Intel Math Kernel Library (MKL),
or the AMD Core Math Library (ACML) as high-speed replacements
for BLAS and LAPACK.  Generally this just involves linking with
the replacement libraries instead of BLAS and LAPACK.

You may need to make minor modifications to "include/armadillo_bits/config.hpp"
in order to make sure Armadillo uses the same style of function names
as used by MKL or ACML. For example, the function names might be in capitals.

On Linux systems, MKL and ACML might be installed in a non-standard
location, such as /opt, which can cause problems during linking.
Before installing Armadillo, the system should know where the MKL or ACML
libraries are located. For example, "/opt/intel/mkl/lib/intel64/".
This can be achieved by setting the LD_LIBRARY_PATH environment variable,
or for a more permanent solution, adding the directory locations
to "/etc/ld.so.conf". It may also be possible to store a text file 
with the locations in the "/etc/ld.so.conf.d" directory.
For example, "/etc/ld.so.conf.d/mkl.conf".
If you modify "/etc/ld.so.conf" or create "/etc/ld.so.conf.d/mkl.conf",
you will need to run "/sbin/ldconfig" afterwards.

Example of the contents of "/etc/ld.so.conf.d/mkl.conf" on a RHEL 6 system,
where Intel MKL version 11.0.3 is installed in "/opt/intel":

/opt/intel/lib/intel64
/opt/intel/mkl/lib/intel64

The default installations of ACML 4.4.0 and MKL 10.2.2.025 are known 
to have issues with SELinux, which is turned on by default in Fedora
(and possibly RHEL). The problem may manifest itself during run-time,
where the run-time linker reports permission problems.
It is possible to work around the problem by applying an appropriate
SELinux type to all ACML and MKL libraries.

If you have ACML or MKL installed and they are persistently giving
you problems during linking, you can disable the support for them
by editing the "CMakeLists.txt" file, deleting "CMakeCache.txt" and
re-running the CMake based installation. Specifically, comment out
the lines containing:
  INCLUDE(ARMA_FindMKL)
  INCLUDE(ARMA_FindACMLMP)
  INCLUDE(ARMA_FindACML)



9: Support for ATLAS
====================

Armadillo can use the ATLAS library for faster versions of
certain LAPACK and BLAS functions. Not all ATLAS functions are
currently used, and as such LAPACK should still be installed.

The minimum recommended version of ATLAS is 3.8.
Old versions (eg. 3.6) can produce incorrect results
as well as corrupting memory, leading to random crashes.

Users of older Ubuntu and Debian based systems should explicitly
check that ATLAS 3.6 is not installed.  It's better to
remove the old version and use the standard LAPACK library.



10: Documentation / API Reference Manual
========================================

A reference manual (documentation of functions and classes) is available at:
  
  http://arma.sourceforge.net/docs.html

The documentation is also in the "docs.html" file in this archive,
which can be viewed with a web browser.



11: MEX Interface to Octave
===========================

The "mex_interface" folder contains examples of how to interface
Octave with C++ code that uses Armadillo matrices.



12: Bug Reports and Frequently Asked Questions
==============================================

Answers to frequently asked questions can be found at:

  http://arma.sourceforge.net/faq.html

This library has gone through extensive testing and
has been successfully used in production environments.
However, as with almost all software, it's impossible
to guarantee 100% correct functionality.

If you find a bug in the library (or the documentation),
we are interested in hearing about it. Please make a
_small_ and _self-contained_ program which exposes the bug,
and then send the program source (as well as the bug description)
to the developers.  The developers' contact details are at:

  http://arma.sourceforge.net/contact.html



13: License
===========

Unless specified otherwise, the Mozilla Public License v2.0 is used.
See the "LICENSE.txt" file for license details.

The file "include/armadillo_bits/fft_engine.hpp" is licensed under
both the Mozilla Public License v2.0 and a 3-clause BSD license.
See the file for license details.

The file "include/armadillo_bits/include_superlu.hpp"
is licensed under both the Mozilla Public License v2.0 and
a 3-clause BSD license.  See the file for license details.



14: Developers and Contributors
===============================

Main sponsoring organisation:
- NICTA
  http://nicta.com.au

Main developers:
- Conrad Sanderson - http://conradsanderson.id.au
- Ryan Curtin      - http://www.ratml.org
- Ian Cullinan
- Dimitrios Bouzas
- Stanislav Funiak

Contributors:
- Matthew Amidon
- Eric R. Anderson
- Kipton Barros
- Benoît Bayol
- Salim Bcoin
- Justin Bedo
- Evan Bollig
- Darius Braziunas
- Filip Bruman
- Ted Campbell
- James Cline
- Chris Cooper
- Clement Creusot
- Chris Davey
- Patrick Dondl
- Alexandre Drouin
- Dirk Eddelbuettel
- Carles Fernandez
- Romain Francois
- Michael McNeil Forbes
- Piotr Gawron
- Charles Gretton
- Franz Gritschneder
- Benjamin Herzog
- Edmund Highcock
- Szabolcs Horvat
- Friedrich Hust
- Ping-Keng Jao
- Jacques-Henri Jourdan
- Yaron Keren
- Kshitij Kulshreshtha
- Oka Kurniawan
- Simen Kvaal
- David Lawrence
- Jussi Lehtola
- Jeremy Mason
- Nikolay Mayorov
- Carlos Mendes
- Sergey Nenakhov
- Artem Novikov
- James Oldfield
- Martin Orlob
- Ken Panici
- Adam Piatyszek
- Jayden Platell
- Vikas Reddy
- Ola Rinta-Koski
- Boris Sabanin
- James Sanders
- Pierre-Andre Savalle
- Alexander Scherbatey
- Gerhard Schreiber
- Ruslan Shestopalyuk
- Shane Stainsby
- Petter Strandmark
- Eric Jon Sundstrom
- Paul Torfs
- Martin Uhrin
- Simon Urbanek
- Unai Uribarri
- Juergen Wiest
- Arnold Wiliem
- Yong Kang Wong
- Buote Xu
- George Yammine
- Sean Young



15: Related Software
====================

* MLPACK: C++ library for machine learning and pattern recognition, built on top of Armadillo.
  http://mlpack.org
  
* libpca: C++ library for principal component analysis
  http://sourceforge.net/projects/libpca/
  
* KL1p: C++ library for sparse recovery of underdetermined linear systems, such as compressed sensing.
  http://kl1p.sourceforge.net  
  
* ArmaNpy: interfaces Armadillo matrices with Python
  http://sourceforge.net/projects/armanpy/

