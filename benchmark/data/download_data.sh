# check existence
if [ -f ./data.zip ] ; then
echo "data.zip has already been downloaded."
fi

# download
if [ ! -f ./data.zip ] ; then
wget --no-check-certificate 'https://box.nju.edu.cn/f/568df86113054a06a0da/?dl=1' -O ./data.zip
fi

# extract files
unzip ./data.zip
echo "benchmark data has been extracted."
