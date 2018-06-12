#!/bin/bash
  
##### INSTALL CMAKE 3.11.3 #####
echo "Installing CMAKE"
wget https://cmake.org/files/v3.11/cmake-3.11.3-Linux-x86_64.sh
sudo sh cmake-3.11.3-Linux-x86_64.sh --prefix=/home/travis/.local/ --exclude-subdir

##### INSTALL CUDA #####
echo "Installing CUDA library"


case ${CUDA:0:3} in

'6.5')
    travis_retry wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_6.5-14_amd64.deb
    travis_retry sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
    ;;
'7.0')
    travis_retry wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
    travis_retry sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
    ;;
'7.5')
    travis_retry wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
    travis_retry sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    ;;
'8.0')
    travis_retry wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    travis_retry sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    ;;
'9.0')
    travis_retry wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    travis_retry sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
    travis_retry sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    ;;
'9.1')
    travis_retry wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
    travis_retry sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
    travis_retry sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    ;;
'9.2')
    travis_retry wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
    travis_retry sudo dpkg -i cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
    travis_retry sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    ;;

esac

travis_retry sudo apt-get update
travis_retry sudo apt-get install cuda -y
travis_retry sudo apt-get clean