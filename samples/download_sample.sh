# check if samples.tar.gz exists
if [ -f ./samples.tar.gz ] ; then
echo "samples.tar.gz has already been downloaded."
fi

# download samples.tar.gz
if [ ! -f ./samples.tar.gz ] ; then
wget --no-check-certificate 'https://box.nju.edu.cn/f/e33302f9b9ce4c7597d0/?dl=1' -O ./samples.tar.gz
fi

# extract files
tar -zxf samples.tar.gz -k
echo "samples have been extracted."
