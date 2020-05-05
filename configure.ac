AC_INIT([kernelPSI], 1.1.0)

AC_LANG(C++)
AC_REQUIRE_CPP

echo "Checking for C++ Compiler"
AC_PROG_CXX

KERNELPSI_CPPFLAGS="-I. "
KERNELPSI_CXXFLAGS=""

if test "${CUDA_HOME}"; then

	AC_MSG_NOTICE("BACKEND = GPU")
	BACKEND="CUDA"

	AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
	if test "${HAS_NVCC}"; then
		AC_MSG_RESULT("NVCC found")
	else
		AC_MSG_ERROR("NVCC not found; check CUDA install")
	fi

	AC_MSG_CHECKING("building the nvcc command line")
	NVCC="${CUDA_HOME}/bin/nvcc"
	AC_SUBST(NVCC)

	AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
	AC_CHECK_FILE([${CUDA_HOME}/lib64/libcudart.so], [BIT64="yes"])
	if test "${BIT64}"; then
		AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
		cu_libdir="${CUDA_HOME}/lib64"
	else
		AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
		cu_libdir="${CUDA_HOME}/lib"
	fi

	AC_MSG_CHECKING("determining GPU architecture")
	CU_ARCH=`python inst//python//arch_detect.py`
	AC_SUBST(CU_ARCH)

	KERNELPSI_LIBS="-L${cu_libdir} -Wl,-rpath,${cu_libdir} -lcudart -lcublas" 
else

	AC_MSG_NOTICE("BACKEND = CPU")
	BACKEND="CPU"
    
	KERNELPSI_LIBS="-Wl,-rpath"
       
fi

## look for Rscript, but use the one found via R_HOME to allow for multiple installations
AC_DEFUN([AC_PROG_RSCRIPT], [AC_CHECK_PROG(RSCRIPT,Rscript,yes)])
AC_PROG_RSCRIPT

if test x"${RSCRIPT}" == x"yes" ; then

    R_HOME=`${R_HOME}/bin/Rscript -e 'cat(R.home())'`
    rcpp_incl=`${R_HOME}/bin/Rscript -e 'Rcpp:::CxxFlags()'`
    rcpparmadillo_incl=`${R_HOME}/bin/Rscript -e 'RcppArmadillo:::CxxFlags()'`
    AC_SUBST(RCPP_INCL,["$rcpp_incl"])
    AC_SUBST(RCPPARMA_INCL,["$rcpparmadillo_incl"])

else
	echo "
  Your installation does not appear to have Rscript installed.

  Please make sure that you have a working and complete R installation.
"
    exit 1
fi

R_INCL=`"${R_HOME}/bin/R" CMD config --cppflags`
R_LIBS=`"${R_HOME}/bin/R" CMD config --ldflags`
R_CPIC=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
CXX="gcc"

AC_MSG_NOTICE([Building Makevars])
AC_SUBST(BACKEND)
AC_SUBST(CXX)
AC_SUBST(KERNELPSI_CPPFLAGS)
AC_SUBST(KERNELPSI_CXXFLAGS)
AC_SUBST(KERNELPSI_LIBS)
AC_SUBST(R_INCL)
AC_SUBST(R_CPIC)
AC_SUBST(R_LIBS)
AC_OUTPUT(src/Makevars)
