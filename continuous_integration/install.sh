#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD


# Travis clone scikit-learn/scikit-learn repository in to a local repository.
# We use a cached directory with three scikit-learn repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip
echo 'download'
ls $HOME/download
echo 'miniconda'
ls $HOME/miniconda

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    pushd .
    cd
    mkdir -p download
    cd download
    echo "Cached in $HOME/download :"
    ls -l
    echo
    if [[ ! -f miniconda.sh ]]
        then
        wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
            -O miniconda.sh
        fi
    chmod +x miniconda.sh
    if [ ! -d "$HOME/miniconda/bin" ]; then
        if [ -d "$HOME/miniconda" ]; then rm -r "$HOME/miniconda"; fi;
        ./miniconda.sh -b
    fi
    cd ..
    export PATH="$HOME/miniconda/bin:$PATH"
    conda update --yes conda
    popd

    # Remove testenv leftover from last build
    conda remove --yes -n testenv --all || \
        echo "No environment 'testenv' to remove";

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION cython=$CYTHON_VERSION
    source activate testenv

    # Install nose-timer via pip
    pip install nose-timer

    # Resolve MKL usage
    if [[ "$INSTALL_MKL" == "true" ]]; then
        conda install --yes mkl
    else
        conda remove --yes --features mkl || echo "MKL not installed"
    fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to used numpy installed through apt-get
    # install.
    deactivate
    # Create a new virtualenv using system site packages for numpy and scipy
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install nose nose-timer
    pip install cython
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

if [ ! -d "$CACHED_BUILD_DIR" ]; then
    mkdir -p $CACHED_BUILD_DIR
fi

rsync -av --exclude '.git/' --exclude='testvenv/' \
      $TRAVIS_BUILD_DIR $CACHED_BUILD_DIR

cd $CACHED_BUILD_DIR/scikit-learn

# Build scikit-learn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop
