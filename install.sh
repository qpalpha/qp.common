#! /bin/bash

echo "=============================== qpc Setup ===============================" 

#%% PYTHONPATH
DEFAULT_PYTHON="/qp/lib/site-package"
read -p "Enter PYTHONPATH: [$DEFAULT_PYTHON]: " pypath
if [ -z $pypath ]
then
    pypath=$DEFAULT_PYTHON
fi 
mkdir -p $pypath
echo -e "export PYTHONPATH=$pypath:\$PYTHONPATH" >> $HOME/.bashrc

#%% Soft link
PWDPATH=`pwd`
read -p "Enter path of qpc: [$PWDPATH]:" eqpath
if [ -z $eqpath ]
then
    eqpath=$PWDPATH
fi 
ln -s $eqpath/qpc.py $pypath/qpc.py

echo "=============================== END ===============================" 
