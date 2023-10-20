# check existence
if [ -f ./data.zip ] ; then
echo "data.zip has already been downloaded."
fi

# download
if [ ! -f ./data.zip ] ; then
wget --no-check-certificate 'https://box.nju.edu.cn/f/5184eb3d6bab458c9aee/?dl=1' -O ./data.zip
fi

# extract files
unzip ./data.zip
echo "benchmark data has been extracted."
