# ========== download lab_pred ==========
# check existence
if [ -f ./lab_pred/lab_pred.zip ] ; then
echo "lab_pred.zip has already been downloaded."
fi

# download
if [ ! -f ./lab_pred/lab_pred.zip ] ; then
mkdir lab_pred
wget --no-check-certificate 'https://box.nju.edu.cn/f/d4e7d4df7b994fd5a63d/?dl=1' -O ./lab_pred/lab_pred.zip
fi

# extract files
unzip -n ./lab_pred/lab_pred.zip -d ./lab_pred/
echo "lab_pred data has been extracted."

# ========== download wild_pred ==========
# check existence
if [ -f ./wild_pred/wild_pred.zip ] ; then
echo "wild_pred.zip has already been downloaded."
fi

# download
if [ ! -f ./wild_pred/wild_pred.zip ] ; then
mkdir wild_pred
wget --no-check-certificate 'https://box.nju.edu.cn/f/c474129c859f438db1fa/?dl=1' -O ./wild_pred/wild_pred.zip
fi

# extract files
unzip -n ./wild_pred/wild_pred.zip -d ./wild_pred/
echo "wild_pred data has been extracted."

