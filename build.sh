export compiler=$(which gcc)
MAJOR=$(echo __GNUC__ | $compiler -E -xc - | tail -n 1)
MINOR=$(echo __GNUC_MINOR__ | $compiler -E -xc - | tail -n 1)
PATCHLEVEL=$(echo __GNUC_PATCHLEVEL__ | $compiler -E -xc - | tail -n 1)
echo == GCC $MAJOR.$MINOR.$PATCHLEVEL

flag=1
if [ $1 ];then
    flag=$1
fi

if [ $flag == 1 ];then
    cmake -B build/ -DTARGET_CXX_SUPPORT:STRING=11
elif [ $flag == 4 ];then
    cmake -B build/ -DTARGET_CXX_SUPPORT:STRING=14
elif [ $flag == 7 ];then
    cmake -B build/ -DTARGET_CXX_SUPPORT:STRING=17
else
    echo "Choose 1(c11) 4(c11) or 7(c17) only!"
    exit 1
fi
cmake --build build/